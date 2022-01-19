import os
from IPython.display import display, HTML, Video

from neurobo.utils.video import write_video


def display_video(
  frames,
  filename='temp.mp4',
  fps=60,
  width=None,
  height=None,
  controls=True,
  autoplay=False,
  muted=False,
  loop=False,
  embed=True,
  verbose=False,
):
  # Write video to a temporary file.
  filepath = os.path.abspath(filename)
  write_video(filepath, frames, fps=fps, verbose=verbose)

  # Read video and display as HTML5 video.
  html_attributes = []
  if controls: html_attributes.append('controls')
  if autoplay: html_attributes.append('autoplay')
  if muted: html_attributes.append('muted')
  if loop: html_attributes.append('loop')
  display(
    Video(
      filepath, embed=embed, width=width, height=height, html_attributes=' '.join(html_attributes)
    )
  )


def display_heading(text, level=1):
  display(HTML('<h{}>{}</h{}>'.format(level, text, level)))
