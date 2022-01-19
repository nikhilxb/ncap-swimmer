import numpy as np
import imageio
import os


def concat_videos(*videos, bg=(255, 255, 255)):
  max_timesteps = max(len(video) for video in videos)
  n_rows = sum(video[0].shape[0] for video in videos)
  n_cols = max(video[0].shape[1] for video in videos)
  frames = [
    np.array([[bg]], dtype=np.uint8).repeat(n_rows, axis=0).repeat(n_cols, axis=1)
    for _ in range(max_timesteps)
  ]
  r, c = 0, 0
  for video in videos:
    dr = video[0].shape[0]
    dc = video[0].shape[1]
    for t, frame in enumerate(video):
      frames[t][r:r + dr, c:c + dc, :] = frame
    r += dr
  return frames


def write_video(filepath, frames, fps=60, macro_block_size=1, verbose=False):
  with imageio.get_writer(filepath, fps=fps, macro_block_size=macro_block_size) as video:
    if verbose: print('Saving video to:', filepath)
    for frame in frames:
      video.append_data(frame)
