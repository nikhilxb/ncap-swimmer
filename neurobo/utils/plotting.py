import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns


def _pt_to_data_units(ax, x, y):
  """Convert (x, y) from point units to data units on axes `ax`."""
  t = ax.transData.inverted()
  return t.transform((x, y)) - t.transform((0, 0))


def _pt_to_axes_units(ax, x, y):
  """Convert (x, y) from point units to axes units on axes `ax`."""
  t = ax.transAxes.inverted()
  return t.transform((x, y)) - t.transform((0, 0))


def _data_to_axes_units(ax, x, y):
  """Convert (x, y) from data units to axes units on axes `ax`."""
  t = ax.transData + ax.transAxes.inverted()
  return t.transform((x, y))


def _clamp(value, value_min=0, value_max=1):
  return max(value_min, min(value, value_max))


def _nonoverlap(coords, x, y, h, width=1, dy=0.1, max_iters=500):
  """Minimizing overlap of labels at (x, y) coordinates.
  Args:
    coords: DataFrame[x, y, ...], with single y per x, with ascending y.
  """
  coords = coords.copy()
  for _ in range(max_iters):
    changed = False
    for i, row_i in coords.iterrows():
      for j, row_j in coords.iterrows():
        if j <= i: continue
        x_i = row_i[x]
        x_j = row_j[x]
        if abs(x_i - x_j) > width: continue
        y_i = row_i[y]
        y_j = row_j[y]
        h_i = row_i[h]
        h_j = row_j[h]
        if abs(y_i - y_j) < (h_i + h_j) / 2:
          # Preserve order when nudging.
          coords.at[i, y] = _clamp(y_i - dy)
          coords.at[j, y] = _clamp(y_j + dy)
          changed = True
    if not changed: break
  return coords


def _linelabels(df, ax, x, y, label, colors=None, loc='end', label_width=100, **kwargs):
  """
  Args:
    df: DataFrame[x, y, label, ...]
    ax: Current axes.
    x: Column for x-axis.
    y: Column for y-axis.
    label: Column for label text.
    colors: Lookup table { label: color }.
    loc: Position of label.
      'end': End of the individual line.
      'right': Right margin of axes.
    label_width: Maximum width of label (in px) used to prevent overlap.
  """
  # Get layout style values from context.
  ctx = sns.plotting_context()
  fontsize = ctx.get('legend.fontsize')

  # Calculate y-positions for labels: Data[x, y, label], with single y per x.
  coords = df[[x, y, label]].groupby([label, x]).mean().reset_index()

  # Need to call `get_xlim/get_ylim` to calculate ax.transLimits properly.
  # Filter values to fit within x-limits so last value is calculated correctly.
  _, xmax = ax.get_xlim()
  _, _ = ax.get_ylim()
  coords = coords.loc[coords[x] <= xmax]
  coords = coords.sort_values([x]).groupby([label]).last().reset_index()
  coords = coords.sort_values([y]).reset_index()
  # Ensure x and y columns are floats to prevent integer truncation when
  # converting to axes units.
  coords = coords.astype({ x: 'float', y: 'float' })

  # Convert (x, y) points from data to axes units.
  text_width, text_height = _pt_to_axes_units(ax, label_width, fontsize)
  for i, row in coords.iterrows():
    ix, iy = _data_to_axes_units(ax, row[x], row[y])
    coords.at[i, x] = _clamp(ix, 1 if loc == 'right' else 0)
    coords.at[i, y] = _clamp(iy)
    coords.at[i, 'height'] = len(row[label].split('\n')) * text_height

  # Prevent overlap along y-axis.
  coords = _nonoverlap(coords, x, y, 'height', width=text_width, dy=text_height / 8)

  # Add text annotation at (x, y) point (in data units).
  for i, row in coords.iterrows():
    text_kwargs = dict(
      # Offset text (in pts) from xy coord.
      xytext=(4, 0),
      textcoords='offset points',
      # Style properties.
      horizontalalignment='left',
      verticalalignment='center',
      linespacing=1,
      fontsize=fontsize,
      color=(colors and colors[row[label]]) or None,
    )
    text_kwargs.update(kwargs)
    a = ax.annotate(
      row[label],
      (row[x], row[y]),
      xycoords='axes fraction',
      **text_kwargs,
    )


def lineplot(
  df,
  x,
  y,
  ax=None,
  xaxis=None,
  yaxis=None,
  xformat=None,
  yformat=None,
  xscale=None,
  yscale=None,
  xlabel=None,
  xlabel_kwargs={},
  ylabel=None,
  ylabel_kwargs={},
  title=None,
  title_kwargs={},
  legend=False,
  legend_loc='lower right',
  legend_cols=1,
  legend_bbox=None,
  legend_kwargs={},
  linelabels=None,
  linelabels_loc='end',  # 'end' | 'right'
  linelabels_kwargs={},
  despine=True,
  **lineplot_kwargs,
):
  if ax is None: fig, ax = plt.subplots()

  # Lineplot.
  assert isinstance(x, str)
  assert isinstance(y, str)
  g = sns.lineplot(data=df, x=x, y=y, ax=ax, **lineplot_kwargs)
  ax.margins(x=0, y=0)

  # Legend.
  if legend:
    ax.legend(loc=legend_loc, ncol=legend_cols, bbox_to_anchor=legend_bbox, **legend_kwargs)
  elif ax.get_legend() is not None:
    ax.get_legend().remove()

  # Axes bounds.
  if xaxis is not None:
    if len(xaxis) == 2:
      x0, x1 = xaxis
      g.set(xlim=(x0, x1))
    elif len(xaxis) == 3:
      x0, x1, dx = xaxis
      g.set(xlim=(x0, x1), xticks=np.arange(x0, x1 + dx, step=dx))
    else:
      raise ValueError(f'Invalid xaxis: {xaxis}')
  if yaxis is not None:
    if len(yaxis) == 2:
      y0, y1 = yaxis
      g.set(ylim=(y0, y1))
    elif len(yaxis) == 3:
      y0, y1, dy = yaxis
      g.set(ylim=(y0, y1), yticks=np.arange(y0, y1 + dy, step=dy))
    else:
      raise ValueError(f'Invalid yaxis: {yaxis}')

  # Axes tick formatter.
  if xformat is not None: ax.xaxis.set_major_formatter(xformat)
  if yformat is not None: ax.yaxis.set_major_formatter(yformat)

  # Axes scale.
  if xscale is not None: ax.set_xscale(xscale)
  if yscale is not None: ax.set_yscale(yscale)

  # Line labels.
  if linelabels is not None:
    assert isinstance(linelabels, str)
    p = sns._core.VectorPlotter(data=df, variables=dict(x=x, y=y, hue=lineplot_kwargs.get('hue')))
    m = sns._core.HueMapping(
      p,
      palette=lineplot_kwargs.get('palette'),
      order=lineplot_kwargs.get('hue_order'),
      norm=lineplot_kwargs.get('hue_norm'),
    )
    _linelabels(
      df,
      ax,
      x,
      y,
      linelabels,
      colors=m.lookup_table,
      loc=linelabels_loc,
      **linelabels_kwargs,
    )

  # Axes title and labels.
  if title is not None: ax.set_title(title, **title_kwargs)
  if xlabel is not None: ax.set_xlabel(xlabel, **xlabel_kwargs)
  if ylabel is not None: ax.set_ylabel(ylabel, **ylabel_kwargs)

  # Borders.
  if despine: sns.despine()
  return g
