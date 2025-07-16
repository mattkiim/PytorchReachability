import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
import io
from PIL import Image, ImageDraw
import numpy as np
import torch
import pickle
import pathlib
import ruamel.yaml as yaml
import os
import sys
import copy
import math

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
dreamer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../dreamerv3-torch'))
sys.path.append(dreamer_dir)
import tools

DEFAULT_VEHICLE_TEMP = 255 / 1.1
MIN_VEHICLE_TEMP = 0. # TODO: implement this

DEFAULT_RGB_VEHICLE_TEMP = 255
MIN_RGB_VEHICLE_TEMP = 255/2
DEFAULT_OBSTACLE_TEMP = 255 / 2 + 0.001

class HeatFrameGenerator:
    def __init__(self, config):
        self.config = config
        self.H = None
        self.W = None
        self.cx = None
        self.cy = None
        self.radius = None
        self.vehicle_temp = DEFAULT_VEHICLE_TEMP
        self.vehicle_temp_rgb = DEFAULT_RGB_VEHICLE_TEMP
        self.vehicle_has_entered = False

    def _compute_geometry(self, img_shape):
        self.H, self.W = img_shape[:2]
        self.cx = int((self.config.obs_x - self.config.x_min) / (self.config.x_max - self.config.x_min) * self.W)
        self.cy = int((self.config.y_max - self.config.obs_y) / (self.config.y_max - self.config.y_min) * self.H)
        self.radius = int(1.1 * self.config.obs_r / (self.config.x_max - self.config.x_min) * self.W)

    def _get_mask(self):
        Y, X = np.ogrid[:self.H, :self.W]
        mask = (X - self.cx)**2 + (Y - self.cy)**2 <= self.radius**2
        mask = mask[:, :, None]
        return mask

    def _get_outline_mask(self):
        Y, X = np.ogrid[:self.H, :self.W]
        d2 = (X - self.cx)**2 + (Y - self.cy)**2
        r1 = self.radius
        return (d2 >= (r1 - 1)**2) & (d2 <= (r1 + 1)**2)
    
    def reset_vehicle_heat(self):
      self.vehicle_temp = DEFAULT_VEHICLE_TEMP
      self.vehicle_temp_rgb = DEFAULT_RGB_VEHICLE_TEMP
      self.vehicle_has_entered = False
    
    def show_heat_image(self, img_heat_array, save_path="test.png"):
      img_heat_array = img_heat_array.squeeze(-1).astype(np.uint8)
      Image.fromarray(img_heat_array).convert("L").save(save_path)

    def get_heat_frame_v0(self, img_array, heat=True):
        '''
        The failure set is the only object with "heat" in the image.
        '''
        heat_frame = img_array[..., 2:3].copy()

        mask = self._get_mask()
        if heat:
            heat_frame[mask] = 0
        else:
            heat_frame[:] = 255

        return heat_frame

    def get_heat_frame_v1(self, img_array, heat=True):
        '''
        The vehicle also has "heat" and becomes dark the instant it 
        enters the unsafe region. 
        '''
        self._compute_geometry(img_array.shape)
        obstacle = img_array[..., 2:3].copy()
        vehicle = img_array[..., 0:1].copy()
        obstacle_mask = self._get_mask()
        
        if heat:
            heat_frame = obstacle
            heat_frame[obstacle_mask] = DEFAULT_OBSTACLE_TEMP

            vehicle_mask = (vehicle == 0)
            heat_frame[vehicle_mask & obstacle_mask] = 0
            heat_frame[vehicle_mask & ~obstacle_mask] = 255 / 4
        else:
            heat_frame = np.ones_like(vehicle) * 255
            vehicle_mask = (vehicle == 0)
            heat_frame[vehicle_mask] = 255 / 4

            # outline_mask = self._get_outline_mask()
            # heat_frame[outline_mask] = 0

        return heat_frame
    
    def get_heat_frame_v2(self, img_array, heat=True, alpha_in=3, alpha_out=5):
        '''
        partial observability
        
        if you spend too long in unsafe, become different color when exiting (RGB)
        the heat map should be the same as v3
        '''
        return self.get_heat_frame_v3(img_array, heat, alpha_in=alpha_in, alpha_out=alpha_out)

    def get_heat_frame_v3(self, img_array, heat=True, alpha_in=3, alpha_out=5, heat_value=None): 
      '''
      full observability
      
      The vehicle becomes darker (lower value) the longer it is inside the obstacle region.
      When it exits the region, it gradually cools down (brightness increases).
      
      alpha_in: how quickly heat accumulates inside the region
      alpha_out: how quickly it fades outside
      '''
      self._compute_geometry(img_array.shape)
      obstacle = img_array[..., 2:3].copy()
      vehicle = img_array[..., 0:1].copy()
      obstacle_mask = self._get_mask()
      
      if heat:
          heat_frame = obstacle.copy()
          heat_frame[obstacle_mask] = DEFAULT_OBSTACLE_TEMP

          vehicle_mask = (
            (vehicle <= DEFAULT_VEHICLE_TEMP) & 
            (vehicle != DEFAULT_OBSTACLE_TEMP)
                          )
          inside_mask = vehicle_mask & obstacle_mask
          outside_mask = vehicle_mask & ~obstacle_mask

          # Apply heat value
          if heat_value is None:
            heat_frame[inside_mask] = self.vehicle_temp
            heat_frame[outside_mask] = self.vehicle_temp
            
            # Update temperature
            if np.any(outside_mask) and not np.any(inside_mask):
                # All vehicle pixels are outside
                self.vehicle_temp = min(DEFAULT_VEHICLE_TEMP, self.vehicle_temp + alpha_out)
                # pass
            elif np.any(inside_mask):
                # Some or all vehicle pixels are inside
                self.vehicle_temp = max(0, self.vehicle_temp - alpha_in)
            
          else:
            temp = self.heat_to_temp(heat_value, DEFAULT_VEHICLE_TEMP)
            temp = np.clip(temp, 0, DEFAULT_VEHICLE_TEMP)
            heat_frame[inside_mask] = temp
            heat_frame[outside_mask] = temp
          
      else:
          heat_frame = np.ones_like(vehicle) * 255
          vehicle_mask = (vehicle == 0)
          heat_frame[vehicle_mask] = DEFAULT_VEHICLE_TEMP

      if config.include_outline:
        # black outline
        from scipy.ndimage import binary_erosion
        vehicle_mask_binary = np.squeeze(vehicle_mask.astype(bool), -1)
        outline = vehicle_mask_binary ^ binary_erosion(vehicle_mask_binary)
        heat_frame = np.squeeze(heat_frame, -1)
        heat_frame[outline] = 0
        heat_frame = heat_frame[..., None]  # Restore original shape (H, W, 1)
        
      return heat_frame, self.vehicle_temp
    
    def heat_to_temp(self, heat_value, def_temp, alpha_in=3):
      temp = -def_temp * (heat_value - 1)
      # print(def_temp, heat_value, temp)
      return temp
    
    def get_rgb_v2(self, img_array, config, heat=True):
        """
        partial observability
        
        Vehicle turns blue when it enters an obstacle.
        When it leaves fully, it stays blue (permanent state change).
        """
        self._compute_geometry(img_array.shape)
        obstacle_mask = self._get_mask()

        # Refined vehicle mask
        vehicle_mask = (
            (img_array[..., 2:3] > 255/2) & 
            (img_array[..., 0:1] < 100) & 
            (img_array[..., 1:2] < 100)
        )

        inside_mask = vehicle_mask & obstacle_mask
        outside_mask = vehicle_mask & ~obstacle_mask

        rgb_out = img_array.copy()

        if heat:
            # Set the flag if vehicle touches the obstacle
            if np.any(inside_mask):
                self.vehicle_has_entered = True

            # If vehicle has entered and fully left, change color
            if self.vehicle_has_entered and not np.any(inside_mask):
                # Apply a permanent color change (e.g., cyan or light blue)
                rgb_out[..., 2:3][vehicle_mask] = 255/2  # Custom color
            else:
                pass

        return rgb_out

    def get_rgb_v3(self, img_array, config, heat=True, alpha_in=10, alpha_out=20, heat_value=None):
        """
        full observability.
        
        Tint the vehicle in the blue channel only.

        vehicle_mask : pixels whose *blue* value is near-zero are considered “vehicle”.
        Heat builds (alpha_in) while they sit inside obstacle_mask
        and cools (alpha_out) when they leave.
        """        
        self._compute_geometry(img_array.shape)
        obstacle_mask = self._get_mask()
        vehicle_mask = (
          (img_array[..., 2:3] > 255/2) & 
          (img_array[..., 0:1] < 100) & 
          (img_array[..., 1:2] < 100)
        )

        inside_mask = vehicle_mask & obstacle_mask
        outside_mask = vehicle_mask & ~obstacle_mask

        rgb_out = img_array.copy()

        if heat:
          if heat_value is None:
            # rgb_out[..., 2:3][inside_mask] = self.vehicle_temp_rgb
            # rgb_out[..., 2:3][outside_mask] = self.vehicle_temp_rgb
            
            temp = self.vehicle_temp_rgb
            temp_norm = temp / DEFAULT_RGB_VEHICLE_TEMP
            decay_factor = temp_norm * 0.4  # decays from 0.4 → 0 as temp goes 0 → 255
            light_blue = np.array([temp * decay_factor, temp * decay_factor, temp])  # R, G, B
            inside_mask = np.squeeze(inside_mask, axis=-1)
            outside_mask = np.squeeze(outside_mask, axis=-1)
            rgb_out[inside_mask] = light_blue
            rgb_out[outside_mask] = light_blue
                    
            if not np.any(inside_mask):
              self.vehicle_temp_rgb = min(DEFAULT_RGB_VEHICLE_TEMP, self.vehicle_temp_rgb + alpha_out * 1.2)
              # pass
            else:
                self.vehicle_temp_rgb = max(MIN_RGB_VEHICLE_TEMP, self.vehicle_temp_rgb - alpha_in * 1.2)
            
          else:
            temp = self.heat_to_temp(heat_value, DEFAULT_RGB_VEHICLE_TEMP)
            temp = np.clip(temp, MIN_RGB_VEHICLE_TEMP, DEFAULT_RGB_VEHICLE_TEMP)
            
            temp_norm = temp / DEFAULT_RGB_VEHICLE_TEMP
            # print(temp, temp_norm, heat_value)
            decay_factor = temp_norm * 0.4  # decays from 0.4 → 0 as temp goes 0 → 255
            light_blue = np.array([temp * decay_factor, temp * decay_factor, temp])  # R, G, B
            inside_mask = np.squeeze(inside_mask, axis=-1)
            outside_mask = np.squeeze(outside_mask, axis=-1)
            rgb_out[inside_mask] = light_blue
            rgb_out[outside_mask] = light_blue
            
            # rgb_out[..., 2:3][inside_mask] = temp
            # rgb_out[..., 2:3][outside_mask] = temp
            
        else:
            temp = self.vehicle_temp_rgb
            decay_factor = 0.4 # no decay when no heat
            light_blue = np.array([temp * decay_factor, temp * decay_factor, temp])  # R, G, B
            inside_mask = np.squeeze(inside_mask, axis=-1)
            outside_mask = np.squeeze(outside_mask, axis=-1)
            rgb_out[inside_mask] = light_blue
            rgb_out[outside_mask] = light_blue
            
            # rgb_out[..., 2:3][inside_mask] = temp
            # rgb_out[..., 2:3][outside_mask] = temp
            
        if config.include_outline:
          # black outline
          from scipy.ndimage import binary_erosion
          mask = np.squeeze(vehicle_mask, -1)
          outline = mask ^ binary_erosion(mask)
          rgb_out[outline] = (0, 0, 0)

        return np.clip(rgb_out, 0, 255).astype(img_array.dtype)

def draw_comet(draw, base_center, angle_rad, length, width, fill_color="blue"):
    """Draw a comet‑shaped arrow whose *base* is centred on ``base_center``.

    The shape is a triangle (tip) + a semicircle (tail) giving a smooth, tapered
    look.  ``angle_rad`` is the *world* heading measured anticlockwise from +x.
    All positions are pixel coordinates *in the current canvas*.
    """
    # PIL's coordinate system has y going *down*; flip the sign so +y in world
    # (up) maps correctly when drawing the comet
    angle_rad = -angle_rad

    tip_x = base_center[0] + length * math.cos(angle_rad)
    tip_y = base_center[1] + length * math.sin(angle_rad)

    radius = width / 2
    # perpendicular for the triangle base
    perp = angle_rad + math.pi / 2
    p1_x = base_center[0] + radius * math.cos(perp)
    p1_y = base_center[1] + radius * math.sin(perp)
    p2_x = base_center[0] - radius * math.cos(perp)
    p2_y = base_center[1] - radius * math.sin(perp)

    # Arrow head (triangle)
    draw.polygon([(tip_x, tip_y), (p1_x, p1_y), (p2_x, p2_y)], fill=fill_color)

    # Semicircle tail
    start_angle_deg = math.degrees(angle_rad) + 90
    end_angle_deg = math.degrees(angle_rad) - 90
    bbox = [
        (base_center[0] - radius, base_center[1] - radius),
        (base_center[0] + radius, base_center[1] + radius),
    ]
    draw.pieslice(bbox, start=start_angle_deg, end=end_angle_deg, fill=fill_color)

def get_frame_pil(states, config, heat_gen, curr_traj_count: int = 0):
    """Return ``img_array_combined, hot, vehicle_temp`` exactly like the original
    Matplotlib implementation, but rendered with **PIL** at high quality.

    * Anti‑aliasing: render at ``scale``× resolution then down‑sample with
      ``Image.Resampling.LANCZOS`` (default ``scale=4``).
    * Obstacle is drawn as a filled red circle.
    * Vehicle is rendered as a blue *comet* arrow via :pyfunc:`draw_comet`.
    """
    # ------------------------------------------------------------------
    # 1. Parameters & helpers
    # ------------------------------------------------------------------
    scale = getattr(config, "aa_scale", 4)
    base_w, base_h = config.size  # final output resolution (width, height)
    hi_w, hi_h = base_w * scale, base_h * scale

    def world_to_px(coord):
        """Map ``(x, y)`` in world space to *hi‑res* pixel coords."""
        x, y = coord
        px = int((x - config.x_min) / (config.x_max - config.x_min) * hi_w)
        # Invert y: world +y is up, pixel +y is down
        py = int((config.y_max - y) / (config.y_max - config.y_min) * hi_h)
        return px, py

    # ------------------------------------------------------------------
    # 2. Create blank hi‑res canvas & Pillow draw context
    # ------------------------------------------------------------------
    img_hi = Image.new("RGB", (hi_w, hi_h), "white")
    draw = ImageDraw.Draw(img_hi)

    # ------------------------------------------------------------------
    # 3. Draw obstacle (filled red circle)
    # ------------------------------------------------------------------
    obs_center_px = world_to_px((config.obs_x, config.obs_y))
    obs_r_px = (config.obs_r / (config.x_max - config.x_min)) * hi_w
    bbox_obs = [
        obs_center_px[0] - obs_r_px,
        obs_center_px[1] - obs_r_px,
        obs_center_px[0] + obs_r_px,
        obs_center_px[1] + obs_r_px,
    ]
    draw.ellipse(bbox_obs, fill=(255, 0, 0), outline=(255, 0, 0))

    # ------------------------------------------------------------------
    # 4. Draw agent (blue comet arrow)
    # ------------------------------------------------------------------
    # Support both 1‑D and batched tensors for *states* (but we only draw the
    # first agent for visualisation, mirroring the original behaviour)
    x = states[0][0] if states.ndim == 2 else states[0]
    y = states[1][0] if states.ndim == 2 else states[1]
    theta = states[2][0] if states.ndim == 2 else states[2]

    base_px = world_to_px((x.item(), y.item()))
    # comet dimensions scale with output size
    comet_len = 17.5 * scale
    comet_w = 10 * scale

    draw_comet(
        draw,
        base_center=base_px,
        angle_rad=theta.item(),
        length=comet_len,
        width=comet_w,
        fill_color="blue",
    )

    # ------------------------------------------------------------------
    # 5. Down‑sample to final resolution (anti‑alias)
    # ------------------------------------------------------------------
    img_lo = img_hi.resize((base_w, base_h), Image.Resampling.LANCZOS)
    img_array = np.array(img_lo)

    # ------------------------------------------------------------------
    # 6. Heat‑frame logic (unchanged)
    # ------------------------------------------------------------------
    hot = False
    vehicle_temp = None
    heat_opt = config.heat_mode

    if config.multimodal:
        hot = curr_traj_count >= config.heat_prop * config.num_trajs

        if heat_opt == 0:
            img_heat_array = heat_gen.get_heat_frame_v0(copy.deepcopy(img_array), heat=hot)
        elif heat_opt == 1:
            img_heat_array = heat_gen.get_heat_frame_v1(copy.deepcopy(img_array), heat=hot)
        elif heat_opt == 2:
            img_heat_array, vehicle_temp = heat_gen.get_heat_frame_v2(
                copy.deepcopy(img_array), heat=hot,
                alpha_in=config.alpha_in, alpha_out=config.alpha_out,
            )
            img_array = heat_gen.get_rgb_v2(copy.deepcopy(img_array), config, heat=hot)
        elif heat_opt == 3:
            img_heat_array, vehicle_temp = heat_gen.get_heat_frame_v3(
                np.array(img_array), heat=hot,
                alpha_in=config.alpha_in, alpha_out=config.alpha_out,
            )
            img_array = heat_gen.get_rgb_v3(
                copy.deepcopy(img_array), config, heat=hot,
                alpha_in=config.alpha_in, alpha_out=config.alpha_out,
            )
        else:
            raise ValueError("Invalid heat_mode")

        img_array_combined = np.concatenate((img_array, img_heat_array), axis=-1)
    else:
        img_array_combined = img_array

    return img_array_combined, hot, vehicle_temp

def get_frame_eval_pil(states, config):
    """Lightweight eval‑time frame generator.

    Visual style is *identical* to :pyfunc:`get_frame` (arrow length, colours,
    anti‑aliasing, obstacle style) but omits the multimodal heat branch and
    therefore returns a plain ``(H,W,3)`` RGB numpy array.
    """
    # -------------------------------------------------------------- parameters
    scale = getattr(config, "aa_scale", 4)
    base_w, base_h = config.size
    hi_w, hi_h = base_w * scale, base_h * scale

    comet_len_px = 17.5 * scale  # fixed length to match training frames
    arrow_width_px = 10 * scale

    # -------------------------------------------------------------- helper
    def world_to_px(coord):
        x, y = coord
        px = int((x - config.x_min) / (config.x_max - config.x_min) * hi_w)
        py = int((config.y_max - y) / (config.y_max - config.y_min) * hi_h)
        return px, py

    # -------------------------------------------------------------- canvas
    img_hi = Image.new("RGB", (hi_w, hi_h), "white")
    draw = ImageDraw.Draw(img_hi)

    # Obstacle
    obs_center_px = world_to_px((config.obs_x, config.obs_y))
    obs_r_px = (config.obs_r / (config.x_max - config.x_min)) * hi_w
    bbox_obs = [
        obs_center_px[0] - obs_r_px,
        obs_center_px[1] - obs_r_px,
        obs_center_px[0] + obs_r_px,
        obs_center_px[1] + obs_r_px,
    ]
    draw.ellipse(bbox_obs, fill=(255, 0, 0), outline=(255, 0, 0))

    # Agent
    x = states[0][0] if states.ndim == 2 else states[0]
    y = states[1][0] if states.ndim == 2 else states[1]
    theta = states[2][0] if states.ndim == 2 else states[2]
    base_px = world_to_px((x.item(), y.item()))
    draw_comet(
        draw,
        base_center=base_px,
        angle_rad=theta.item(),
        length=comet_len_px,
        width=arrow_width_px,
        fill_color="blue",
    )

    # Anti‑alias
    img_lo = img_hi.resize((base_w, base_h), Image.Resampling.LANCZOS)

    return np.array(img_lo)

def get_frame(states, config, heat_gen, curr_traj_count=0):
  dt = config.dt
  v = config.speed
  fig,ax = plt.subplots()
  plt.xlim([config.x_min, config.x_max])
  plt.ylim([config.y_min, config.y_max])
  plt.axis('off')
  fig.set_size_inches(1, 1)
  # Create the circle patch
  circle = patches.Circle([config.obs_x, config.obs_y], config.obs_r, edgecolor=(1,0,0), facecolor=(1,0,0))
  # Add the circle patch to the axis
  ax.add_patch(circle)
  plt.quiver(states[0], states[1], dt*v*torch.cos(states[2]), dt*v*torch.sin(states[2]), angles='xy', scale_units='xy', minlength=0,width=0.1, scale=0.18,color=(0,0,1), zorder=3)
  plt.scatter(states[0], states[1],s=20, color=(0,0,1), zorder=3)
  plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

  buf = io.BytesIO()
  plt.savefig(buf, format='png', dpi=config.size[0])
  buf.seek(0)

  # load the buffer content as an RGB image
  img = Image.open(buf).convert('RGB')
  img_array = np.array(img)
  
  # generate heat frame of image
  hot = False
  heat_opt = config.heat_mode
  # heat_gen = HeatFrameGenerator(config, last_heat_frame)
  
  if config.multimodal:
    hot = curr_traj_count >= config.heat_prop * config.num_trajs
    if heat_opt == 0:
      img_heat_array = heat_gen.get_heat_frame_v0(copy.deepcopy(img_array), heat=hot)
    elif heat_opt == 1:
      img_heat_array = heat_gen.get_heat_frame_v1(copy.deepcopy(img_array), heat=hot)
    elif heat_opt == 2:
      img_heat_array, vehicle_temp = heat_gen.get_heat_frame_v2(copy.deepcopy(img_array), heat=hot, alpha_in=config.alpha_in, alpha_out=config.alpha_out)
      img_array = heat_gen.get_rgb_v2(copy.deepcopy(img_array), config, heat=hot)
    elif heat_opt == 3:
      img_heat_array, vehicle_temp = heat_gen.get_heat_frame_v3(np.array(img_array), heat=hot, alpha_in=config.alpha_in, alpha_out=config.alpha_out)
      img_array = heat_gen.get_rgb_v3(copy.deepcopy(img_array), config, heat=hot, alpha_in=config.alpha_in, alpha_out=config.alpha_out)
    else:
      raise ValueError("Invalid heat_mode")
      
    img_array_combined = np.concatenate((img_array, img_heat_array), axis=-1)
  else:
    img_array_combined = img_array
  
  plt.close(fig)
  return img_array_combined, hot, vehicle_temp
  
def get_frame_eval(states, config):
  dt = config.dt
  v = config.speed
  fig,ax = plt.subplots()
  plt.xlim([config.x_min, config.x_max])
  plt.ylim([config.y_min, config.y_max])
  plt.axis('off')
  fig.set_size_inches(1, 1)
  # Create the circle patch
  circle = patches.Circle([config.obs_x, config.obs_y], config.obs_r, edgecolor=(1,0,0), facecolor=(1,0,0))
  # Add the circle patch to the axis
  ax.add_patch(circle)
  plt.quiver(states[0], states[1], dt*v*torch.cos(states[2]), dt*v*torch.sin(states[2]), angles='xy', scale_units='xy', minlength=0, width=0.1, scale=0.18, color=(0,0,1), zorder=3)
  plt.scatter(states[0], states[1],s=20, color=(0,0,1), zorder=3)
  plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

  buf = io.BytesIO()
  plt.savefig(buf, format='png', dpi=config.size[0])
  buf.seek(0)

  # load the buffer content as an RGB image
  img = Image.open(buf).convert('RGB')
  img_array = np.array(img)
  
  plt.close(fig)
  return img_array

def get_init_state(config):
  # don't sample inside the failure set
  states = torch.zeros(3)
  while np.linalg.norm(states[:2] - np.array([config.obs_x, config.obs_y])) < config.obs_r:
    states = torch.rand(3)
    
    states[0] *= (config.x_max-config.buffer) - (config.x_min + config.buffer)
    states[1] *= (config.y_max-config.buffer) - (config.y_min + config.buffer)
    states[0] += config.x_min + config.buffer
    states[1] += config.y_min + config.buffer

  # so that the trajectory doesn't immediately go out of bounds
  states[2] = torch.atan2(-states[1], -states[0]) + np.random.normal(0, 1)
  states[2] = states[2] % (2*np.pi)
  
  if config.test:
    states[0] = -0.8 # TODO: comment out
    states[1] = 0.
    states[2] = 0.
    
  return states

def gen_one_traj_img(config, curr_traj_count=0):
  states = get_init_state(config)
  
  state_obs = []
  img_obs = []
  heat_obs = []
  heat_gt = []
  state_gt = []
  dones = []
  acs = []
  u_max = final_config.turnRate
  dt = config.dt
  v = config.speed

  heat_gen = HeatFrameGenerator(config)
  heat_gen.reset_vehicle_heat()
  
  for t in range(config.data_length):
    # random between -u_max and u_max
    ac = torch.rand(1) * 2 * u_max - u_max
    if config.test:
      ac = torch.tensor(0)
    
    # dubin's dynamics
    states_next = torch.rand(3)
    states_next[0] = states[0] + v*dt*torch.cos(states[2])
    states_next[1] = states[1] + v*dt*torch.sin(states[2])
    states_next[2] = states[2] + dt*ac

    # the data is (o_t, a_t), don't observe o_t+1 yet
    if t == config.data_length-1:
      dones.append(1)
    elif torch.abs(states[0]) > config.x_max-config.buffer or torch.abs(states[1]) > config.y_max-config.buffer: 
      # handle out of bounds
      dones.append(1)
    else:
      dones.append(0)
      
    acs.append(ac)
    
    if config.use_pil:
      img_array, hot, vehicle_temp = get_frame_pil(states, config, heat_gen, curr_traj_count=curr_traj_count)
    else:
      img_array, hot, vehicle_temp = get_frame(states, config, heat_gen, curr_traj_count=curr_traj_count)
    norm_temp = 1. - vehicle_temp / DEFAULT_VEHICLE_TEMP
    
    state_obs.append(states[2].numpy())
    # print(state_obs); quit()
    
    state_gt.append(states.numpy()) # gt state for debugging
    if config.multimodal: 
      img_obs.append(img_array[..., :3]) # TODO: turn into dict and grab
      # print(img_array[..., -1:].mean()); quit()
      heat_obs.append(img_array[..., -1:])
      # print(vehicle_temp / DEFAULT_VEHICLE_TEMP)
      heat_gt.append(norm_temp) # TODO: store this in a variable
    else: 
      img_obs.append(img_array)
      heat_obs.append(img_array[..., 0] * 0)
      heat_gt.append(0.0) # TODO: should also be vehicle_temp / DEFAULT_VEHICLE_TEMP
    states = states_next
    if dones[-1] == 1:
      break
  return state_obs, acs, state_gt, img_obs, heat_obs, heat_gt, dones

def generate_trajs(config):
  demos = []
  curr_traj_count = 0
  for i in range(config.num_trajs):
    state_obs, acs, state_gt, img_obs, heat_obs, heat_gt, dones = gen_one_traj_img(config, curr_traj_count=curr_traj_count)
    # print(np.mean(img_obs), np.mean(heat_obs)); quit()
    demo = {}
    demo['obs'] = {'image': img_obs, 'heat': heat_obs, 'priv_heat': heat_gt, 'state': state_obs, 'priv_state': state_gt}
    demo['actions'] = acs
    demo['dones'] = dones
    demos.append(demo)
    print('demo: ', i, "timesteps: ", len(state_obs))
    curr_traj_count += 1

  if config.multimodal:
    with open(f"{config.dataset_path}_{config.alpha_in}" + ".pkl", 'wb') as f: # TODO: read from config
      pickle.dump(demos, f)
  else:
    with open('wm_demos' + str(config.size[0]) + '.pkl', 'wb') as f:
      pickle.dump(demos, f)

def recursive_update(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and key in base:
            recursive_update(base[key], value)
        else:
            base[key] = value

if __name__=='__main__':      
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="configs.yaml", type=str)
    config, remaining = parser.parse_known_args()

    yaml = yaml.YAML(typ="safe", pure=True)
    configs = yaml.load(
        (pathlib.Path(sys.argv[0]).parent / f"../{config.config_path}").read_text()
    )

    name_list = ["defaults"]

    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    final_config = parser.parse_args(remaining)

    demos = generate_trajs(final_config)
