from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, HoverTool, TapTool, CustomJS
from bokeh.layouts import column
from bokeh.models.widgets import Div

from vali_objects.scoring.scoring import Scoring
from vali_objects.utils.logger_utils import LoggerUtils
from vali_objects.utils.subtensor_weight_setter import SubtensorWeightSetter
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils

if __name__ == "__main__":
    logger = LoggerUtils.init_logger("run incentive review")

    subtensor_weight_setter = SubtensorWeightSetter(None, None, None)

    hotkeys = ValiBkpUtils.get_directories_in_dir(ValiBkpUtils.get_miner_dir())

    eliminations_json = ValiUtils.get_vali_json_file(
        ValiBkpUtils.get_eliminations_dir()
    )["eliminations"]

    returns_per_netuid = subtensor_weight_setter.calculate_return_per_netuid(
        local=True, hotkeys=hotkeys, eliminations=eliminations_json
    )

    filtered_results = [(hotkeys[k], v) for k, v in returns_per_netuid.items()]
    scaled_transformed_list = Scoring.transform_and_scale_results(filtered_results)

    sorted_data = sorted(scaled_transformed_list, key=lambda x: x[1], reverse=True)

    y_values = [item[1] for item in sorted_data]
    top_miners = [x[0] for x in sorted_data]

    logger.info(f"top miners [{top_miners}]")
    logger.info(f"top miners scores [{y_values}]")

    # Create a ColumnDataSource with the data
    source = ColumnDataSource(data=dict(x=list(range(len(y_values))), y=y_values, miners=top_miners, text_visible=[False] * len(y_values)))

    # Create a separate ColumnDataSource for the specific miner
    specific_miner_id = "5GhCxfBcA7Ur5iiAS343xwvrYHTUfBjBi4JimiL5LhujRT9t"
    specific_miner_index = top_miners.index(specific_miner_id)
    specific_miner_source = ColumnDataSource(data=dict(x=[specific_miner_index], y=[y_values[specific_miner_index]], miners=[specific_miner_id]))

    # Create a Bokeh figure with fixed y-axis range
    p = figure(title="Top Miner Incentive - Reduced Omega", x_axis_label="Miner Index", y_axis_label="Incentive Score",
               sizing_mode="stretch_both", y_range=(0, 0.1))

    # Add a circle glyph with hover tooltips for all miners
    hover = HoverTool(tooltips=[("Miner", "@miners"), ("Score", "@y")])
    p.add_tools(hover)
    p.circle("x", "y", source=source, size=10, color="blue", alpha=0.7, legend_label="Miners")

    # Add a circle glyph for the specific miner
    p.circle("x", "y", source=specific_miner_source, size=12, color="orange", alpha=0.7, legend_label="Specific Miner")

    # Add labels for each data point, positioned above and to the right
    labels = p.text("x", "y", text="miners", source=source, text_align="left", text_baseline="bottom",
                    text_font_size="8pt", x_offset=5, y_offset=5)

    # Create a TapTool to toggle label visibility on click
    tap_tool = TapTool(renderers=[labels])
    p.add_tools(tap_tool)

    # Define a JavaScript callback to update label visibility on tap
    tap_callback = CustomJS(args=dict(source=source), code="""
        const text_visible = source.data.text_visible;
        const index = cb_data.source.selected.indices[0];
        text_visible[index] = !text_visible[index];
        source.change.emit();
    """)
    tap_tool.callback = tap_callback

    # Add a legend
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"

    # Create a Div to display additional information
    div = Div(text=f"<b>Top Miners:</b> {', '.join(top_miners)}<br><b>Scores:</b> {', '.join(map(str, y_values))}")

    # Create a layout with the plot and the Div
    layout = column(p, div, sizing_mode="stretch_both")

    # Output the layout to an HTML file
    output_file("top_miners_incentive.html")

    # Show the layout
    show(layout)