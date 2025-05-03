import { app } from "/scripts/app.js";
import { ComfyWidgets } from "/scripts/widgets.js";
import { api } from "/scripts/api.js";
import { $el } from "/scripts/ui.js";


// Reference to the original link rendering function
const original_renderLink = LGraphCanvas.prototype.renderLink;

// Custom link rendering function
function renderCustomLink(ctx, a, b, link, skip_border, flow_color, link_color, isSelected, isExecuting) {
    // a: starting point [x, y]
    // b: ending point [x, y]
    // link: link object containing info like origin_id, target_id, type, etc.
    // skip_border: boolean
    // flow_color: color for executing flow animation
    // link_color: base color of the link
    // isSelected: boolean
    // isExecuting: boolean

    const start_node = app.graph.getNodeById(link.origin_id);
    const end_node = app.graph.getNodeById(link.target_id);
    const start_slot = start_node.outputs[link.origin_slot];
    const end_slot = end_node.inputs[link.target_slot];

    // --- Your Custom Logic Goes Here ---
    // Calculate deltaX and deltaY
    const deltaX = Math.abs(a[0] - b[0]);
    const deltaY = Math.abs(a[1] - b[1]);

    // Example: Determine curve radius based on deltas (needs refinement)
    // This is a placeholder - the actual math needs to implement the exponential curve idea
    const min_radius = 5;
    const max_radius = 50;
    // A simple inverse relationship - smaller distance -> larger radius (needs adjustment for desired effect)
    // You might want a function where radius decreases as the ratio deltaX/deltaY (or vice versa) gets smaller
    let radius = max_radius - (deltaX + deltaY) * 0.1; // Very basic placeholder
    radius = Math.max(min_radius, Math.min(max_radius, radius));

    // Placeholder: Draw a simple straight line for now
    // Replace this with ctx drawing commands (bezierCurveTo, lineTo, etc.)
    // to create the axis-aligned path with curved corners using the calculated radius.
    ctx.save();
    ctx.lineWidth = 3;
    ctx.strokeStyle = link_color || "#AAA"; // Use provided link color or default
    if (isSelected) {
        ctx.strokeStyle = "#FFF"; // Highlight selected links
    }
    ctx.beginPath();
    ctx.moveTo(a[0], a[1]);
    // TODO: Implement the actual path drawing logic using radius
    // Example: Move halfway horizontally, curve, move vertically, curve, move horizontally
    ctx.lineTo(b[0], b[1]); // Simple straight line placeholder
    ctx.stroke();
    ctx.restore();

    // --- End Custom Logic ---

    // Optional: Call original function for border/flow if needed, or replicate its logic
    // original_renderLink.call(this, ctx, a, b, link, skip_border, flow_color, link_color, isSelected, isExecuting);
}


// Add the new link type option
// Need to find the correct way to add to LiteGraph link render modes
// This might involve patching LiteGraph or finding a ComfyUI specific mechanism.
// For now, we'll focus on overriding the drawing function.

// Override the main link rendering function
LGraphCanvas.prototype.renderLink = function(ctx, a, b, link, skip_border, flow_color, link_color, isSelected, isExecuting) {
    // Check if the graph's link_render_mode is set to our custom type
    // We need a way to *set* this mode first. Let's assume a mode number like 3 or a string 'CUSTOM'
    const customModeIdentifier = 'CUSTOM_EXPONENTIAL'; // Example identifier

    if (this.link_render_mode === LiteGraph.CUSTOM_EXPONENTIAL) { // Assuming we can add this constant
        renderCustomLink.call(this, ctx, a, b, link, skip_border, flow_color, link_color, isSelected, isExecuting);
    } else {
        // Call the original function for other modes (SPLINE, LINEAR, STRAIGHT)
        original_renderLink.call(this, ctx, a, b, link, skip_border, flow_color, link_color, isSelected, isExecuting);
    }
};


app.registerExtension({
	name: "my_custom_nodepack.CustomLinkStyle",
	async setup(app) {
		// Add the custom render mode to LiteGraph if possible
        // This is the tricky part - LiteGraph doesn't have a built-in way to register new modes easily.
        // We might need to add it directly or find a ComfyUI hook.
        if (LiteGraph) {
            // Add an identifier for our mode. Using a high number or string.
            LiteGraph.CUSTOM_EXPONENTIAL = 4; // Or potentially a string if supported?
            // We also need to add an option to the settings menu to select this mode.
            // This usually involves modifying the settings dialog creation.

            console.log("Custom link style extension: Attempting to add CUSTOM_EXPONENTIAL mode.");

            // Monkey-patch the settings dialog to include the custom link render mode
            const originalShowSettings = ComfyWidgets.showSettings;
            ComfyWidgets.showSettings = function() {
                originalShowSettings();
                setTimeout(() => {
                    const select = document.querySelector('select[name="link_render_mode"]');
                    if (select && !select.querySelector(`option[value="${LiteGraph.CUSTOM_EXPONENTIAL}"]`)) {
                        select.add(new Option('Custom Exponential', LiteGraph.CUSTOM_EXPONENTIAL));
                    }
                }, 100);
            };

            // Monkey-patch ComfyWidgets.createCombo to inject our custom option
            const origCreateCombo = ComfyWidgets.createCombo;
            ComfyWidgets.createCombo = function(name, options, defaultValue, onChange) {
                if (name === "link_render_mode") {
                    options.push({ text: "Custom Exponential", value: LiteGraph.CUSTOM_EXPONENTIAL });
                }
                return origCreateCombo.call(this, name, options, defaultValue, onChange);
            };
        }
	},
    async loadedGraphNode(node, app) {
        // Example: Could potentially modify node rendering based on links, but not needed for link style itself.
    },
    // Add other hooks if needed
});

console.log("Custom link style extension loaded.");
