import os
import re
import base64
import io
import pandas as pd

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.plotting import figure
from bokeh.models import (
    ColumnDataSource, FileInput, MultiSelect,
    Dropdown, Button, Range1d, LinearAxis, Div
)

# -------------------------------
# Configuration
# -------------------------------
SRC_PATH = os.getcwd()
ROOT_PATH = os.path.abspath(os.path.join(SRC_PATH, "../"))
colors = ["blue", "green", "red", "orange", "purple", "brown", "pink", "gray", "cyan", "lime", "magenta"]

# -------------------------------
# Data store for uploaded CSVs
# -------------------------------
# maps (signal_code, ID, nr) -> DataFrame
uploaded_dfs = {}

# -------------------------------
# Create the Bokeh plot
# -------------------------------
p = figure(
    title="Vrednosti signala skozi čas",
    x_axis_label='Čas[s]', y_axis_label='Vrednost signala',
    width=1200
)

# second y-range for anxiety values
p.extra_y_ranges = {"right": Range1d(start=0, end=1)}
p.add_layout(LinearAxis(y_range_name="right", axis_label="Vrednost anksioznosti"), 'right')

# -------------------------------
# FileInput widget for CSV upload
# -------------------------------
file_input = FileInput(accept=".csv", multiple=True, width=400)
file_list_div = Div(text="No files uploaded yet", width=400)

# -------------------------------
# Selection widgets for plotting
# -------------------------------
available_signals = ['021', '022', '023', '024', '025', '026', '027']
multi_select_signals = MultiSelect(options=available_signals)

available_IDs = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011']
multi_select_IDs = MultiSelect(options=available_IDs)

available_nrs = ['001', '002', '003', '004', '005']
menu_nrs = [(noe, noe) for noe in available_nrs]
dropdown_nrs = Dropdown(label="Izberi št. poskusa", menu=menu_nrs)

button = Button(label="Izriši")

# -------------------------------
# State variables for selections
# -------------------------------
selected_signals = []
selected_IDs = []
selected_nr = None

# -------------------------------
# Callbacks for selection widgets
# -------------------------------
def update_signals(attr, old, new):
    global selected_signals
    selected_signals = new

multi_select_signals.on_change('value', update_signals)


def update_IDs(attr, old, new):
    global selected_IDs
    selected_IDs = new

multi_select_IDs.on_change('value', update_IDs)


def update_nr(event):
    global selected_nr
    selected_nr = event.item
    dropdown_nrs.label = f"Izbran: {selected_nr}"

dropdown_nrs.on_click(update_nr)

# -------------------------------
# Callback to handle file uploads
# -------------------------------
def upload_files(attr, old, new):
    # new is list of filenames; file_input.value holds list of base64 strings
    names = file_input.filename
    values = file_input.value
    if not names or not values:
        file_list_div.text = "No files uploaded yet"
        return
    if isinstance(names, str):
        names = [names]
    if isinstance(values, str):
        values = [values]

    # clear previous uploads
    uploaded_dfs.clear()
    items_html = []

    for fname, b64 in zip(names, values):
        try:
            decoded = base64.b64decode(b64)
            bio = io.BytesIO(decoded)
            df = pd.read_csv(bio, header=None)
        except Exception as e:
            print(f"Failed to parse {fname}: {e}")
            continue

        m = re.match(r'^(\d+)-(\d+)-(\d+)-(\d+)\.csv$', fname)
        if m:
            signal_code, id_code, nr_code, suffix = m.groups()
            uploaded_dfs[(signal_code, id_code, nr_code)] = df
            items_html.append(f"<li>{fname}</li>")
        else:
            print(f"Filename {fname} doesn't match expected pattern.")

    file_list_div.text = "<b>Files uploaded</b><ul>"

file_input.on_change('filename', upload_files)

# -------------------------------
# Draw data based on uploads and selections
# -------------------------------
from bokeh.models import ColumnDataSource
from bokeh.models import Marker


def draw_data():
    if not selected_signals or not selected_IDs or selected_nr is None:
        p.title.text = "Ena ali več možnosti v spustnem seznamu ni izbranih!"
        return

    # clear previous renderers
    p.renderers = []
    if p.legend:
        p.legend.items = []

    color_index = 0
    for signal in selected_signals:
        for sel_id in selected_IDs:
            key = (signal, sel_id, selected_nr)
            # main signal
            if key in uploaded_dfs:
                df_main = uploaded_dfs[key]
                min_val, max_val = df_main[3].min(), df_main[3].max()
                norm = (df_main[3] - min_val) / (max_val - min_val)
                cds_main = ColumnDataSource({'time': df_main[2], 'signal': norm})
                color = colors[color_index % len(colors)]
                p.line(
                    'time', 'signal', source=cds_main,
                    legend_label=f"Signal {signal}-{sel_id}-{selected_nr}",
                    line_width=2, color=color
                )
                color_index += 1
            else:
                print(f"No upload for signal file {signal}-{sel_id}-{selected_nr}-000.csv")

            # anxiety signal = code '020'
            key020 = ('020', sel_id, selected_nr)
            if key020 in uploaded_dfs:
                df_020 = uploaded_dfs[key020]
                cds_020 = ColumnDataSource({'time': df_020[2], 'signal': df_020[4]})
                p.scatter(
                    'time', 'signal', source=cds_020,
                    legend_label=f"Anks 020-{sel_id}-{selected_nr}",
                    marker='square', size=8,
                    y_range_name='right',
                    fill_alpha=1, line_color='black', line_width=1,
                    color=colors[(color_index-1) % len(colors)]
                )
            else:
                print(f"No upload for anks file 020-{sel_id}-{selected_nr}-000.csv")

    # reset ranges
    p.y_range.start, p.y_range.end = 0, 1
    p.extra_y_ranges['right'].start, p.extra_y_ranges['right'].end = 0, 1
    p.title.text = "Vrednosti signala skozi čas"

button.on_click(draw_data)

# -------------------------------
# Layout and add to document
# -------------------------------
controls = row(
    column(Div(text="<b>Izberi datoteke</b>"), file_input, file_list_div),
    column(Div(text="<b>Izbira signala</b>"), multi_select_signals),
    column(Div(text="<b>Izbira uID</b>"), multi_select_IDs),
    column(Div(text="<b>Izbira št. poskusa</b>"), dropdown_nrs, button)
)
layout = column(controls, p)

curdoc().add_root(layout)
curdoc().title = "FileInput Data Viewer"
