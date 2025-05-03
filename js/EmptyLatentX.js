import { app } from "../../scripts/app.js";

class EmptyLatentX {
    static BUTTON_BASE_SIZE = 28;
    static BUTTON_LONG_SIDE = 38;
    static BUTTON_MARGIN = 10;
    static BUTTONS_Y = 180;
    static PRESET_BUTTON_WIDTH = 64;
    static PRESET_BUTTON_HEIGHT = 24;
    static PRESETS_PER_ROW = 4;

    constructor(node) {
        this.node = node;
        this.node.size = [306, 315];
        this.squareSize = 30;
        this.isActive = false;

        this.widthWidget = this.node.widgets.find(w => w.name === "width");
        this.heightWidget = this.node.widgets.find(w => w.name === "height");

        this.initializeData();
        this.setupWidgetCallbacks();
        this.setupNodeEventHandlers();
    }

    initializeData() {
        this.aspectRatios = {
            "1:1": { x: 1, y: 1 },
            "2:3": { x: 2, y: 3 },
            "3:2": { x: 3, y: 2 },
            "3:4": { x: 3, y: 4 },
            "4:3": { x: 4, y: 3 },
            "9:16": { x: 9, y: 16 },
            "16:9": { x: 16, y: 9 }
        };

        this.presets = [
            { width: 1024, height: 1024 },
            { width: 896, height: 1152 },
            { width: 896, height: 1216 },
            { width: 832, height: 1152 },
            { width: 832, height: 1216 },
            { width: 1152, height: 832 },
            { width: 1152, height: 896 },
            { width: 1216, height: 896 }
        ];

        this.colors = {
            normal: "rgba(100, 100, 100, 0.8)",
            hover: "rgba(150, 150, 150, 0.9)",
            active: "rgba(0, 100, 200, 0.8)",
            activeHover: "rgba(30, 130, 230, 0.8)"
        };

        this.activeAspect = null;
        this.hoveredAspect = null;
        this.hoveredPreset = null;
        this.presetMargin = 10;
    }

    setupWidgetCallbacks() {
        if (this.widthWidget) {
            this.setupWidgetCallback(this.widthWidget, 'width');
        }
        if (this.heightWidget) {
            this.setupWidgetCallback(this.heightWidget, 'height');
            this.setupHeightChangeHandler();
        }
    }

    setupWidgetCallback(widget, type) {
        const originalCallback = widget.callback;
        widget.callback = (value) => {
            value = this.roundToGrid(value);
            if (this.activeAspect) {
                this.updateLinkedDimension(type, value);
            }
            return originalCallback ? originalCallback.call(widget, value) : value;
        };

        widget.onRemoteChange = (value) => {
            value = this.roundToGrid(value);
            if (this.activeAspect) {
                this.updateLinkedDimension(type, value);
                this.node.setDirtyCanvas(true);
            }
        };
    }

    roundToGrid(value) {
        return Math.round(value / 64) * 64;
    }

    updateLinkedDimension(type, value) {
        const aspect = this.aspectRatios[this.activeAspect];
        let newValue;

        if (type === 'width' && this.heightWidget) {
            newValue = Math.round((value / aspect.x) * aspect.y);
            // Округляем до ближайшего большего числа, кратного 64
            if (newValue % 64 !== 0) {
                newValue = Math.ceil(newValue / 64) * 64;
            }
            this.heightWidget.value = newValue;
        } else if (type === 'height' && this.widthWidget) {
            newValue = Math.round((value / aspect.y) * aspect.x);
            // Округляем до ближайшего большего числа, кратного 64
            if (newValue % 64 !== 0) {
                newValue = Math.ceil(newValue / 64) * 64;
            }
            this.widthWidget.value = newValue;
        }
    }

    setupHeightChangeHandler() {
        this.heightWidget.onchange = () => {
            if (this.activeAspect) {
                this.updateLinkedDimension('height', this.heightWidget.value);
            }
        };
    }

    handleMouseMove(e) {
        const mouseX = e.canvasX - this.node.pos[0];
        const mouseY = e.canvasY - this.node.pos[1];

        const prevHoveredAspect = this.hoveredAspect;
        const prevHoveredPreset = this.hoveredPreset;

        this.hoveredAspect = this.getHoveredAspect(mouseX, mouseY);
        this.hoveredPreset = this.getHoveredPreset(mouseX, mouseY);

        if (prevHoveredAspect !== this.hoveredAspect ||
            prevHoveredPreset !== this.hoveredPreset) {
            this.node.setDirtyCanvas(true);
        }
    }

    handleAspectClick(ratio) {
        if (this.activeAspect === ratio) {
            this.activeAspect = null;
        } else {
            this.activeAspect = ratio;
            if (this.widthWidget && this.heightWidget) {
                const aspect = this.aspectRatios[ratio];
                const width = this.widthWidget.value;
                let height = Math.round((width / aspect.x) * aspect.y);
                height = Math.ceil(height / 64) * 64;
                this.heightWidget.value = height;
            }
        }
        this.node.setDirtyCanvas(true);
    }

    handlePresetClick(index) {
        this.activeAspect = null;
        const preset = this.presets[index];

        if (this.widthWidget && this.heightWidget) {
            this.widthWidget.value = preset.width;
            this.heightWidget.value = preset.height;
        }
        this.node.setDirtyCanvas(true);
    }

    setupNodeEventHandlers() {
        this.node.onMouseMove = this.handleMouseMove.bind(this);
        this.node.onMouseDown = this.handleMouseDown.bind(this);
        this.node.onDrawForeground = this.drawForeground.bind(this);
        this.node.onWidgetChange = this.handleWidgetChange.bind(this);
    }

    getHoveredAspect(mouseX, mouseY) {
        let x = EmptyLatentX.BUTTON_MARGIN;
        for (const [ratio, aspect] of Object.entries(this.aspectRatios)) {
            const { width: buttonWidth, height: buttonHeight } =
                this.getButtonDimensions(ratio, aspect);

            const buttonY = EmptyLatentX.BUTTONS_Y +
                (EmptyLatentX.BUTTON_BASE_SIZE - buttonHeight) / 2;

            if (this.isPointInButton(mouseX, mouseY, x, buttonY, buttonWidth, buttonHeight)) {
                return ratio;
            }
            x += buttonWidth + EmptyLatentX.BUTTON_MARGIN;
        }
        return null;
    }

    getHoveredPreset(mouseX, mouseY) {
        const lineY = EmptyLatentX.BUTTONS_Y + EmptyLatentX.BUTTON_BASE_SIZE + 20;
        const presetsStartY = lineY + 20;

        for (let i = 0; i < this.presets.length; i++) {
            const row = Math.floor(i / EmptyLatentX.PRESETS_PER_ROW);
            const col = i % EmptyLatentX.PRESETS_PER_ROW;

            const presetX = this.presetMargin +
                col * (EmptyLatentX.PRESET_BUTTON_WIDTH + this.presetMargin);
            const presetY = presetsStartY +
                row * (EmptyLatentX.PRESET_BUTTON_HEIGHT + this.presetMargin);

            if (this.isPointInButton(
                mouseX, mouseY, presetX, presetY,
                EmptyLatentX.PRESET_BUTTON_WIDTH,
                EmptyLatentX.PRESET_BUTTON_HEIGHT
            )) {
                return i;
            }
        }
        return null;
    }

    isPointInButton(x, y, buttonX, buttonY, width, height) {
        return x >= buttonX && x <= buttonX + width &&
               y >= buttonY && y <= buttonY + height;
    }

    getButtonDimensions(ratio, aspect) {
        if (ratio === "1:1") {
            return {
                width: EmptyLatentX.BUTTON_BASE_SIZE,
                height: EmptyLatentX.BUTTON_BASE_SIZE
            };
        }
        return {
            width: aspect.x > aspect.y ? EmptyLatentX.BUTTON_LONG_SIDE : EmptyLatentX.BUTTON_BASE_SIZE,
            height: aspect.x > aspect.y ? EmptyLatentX.BUTTON_BASE_SIZE : EmptyLatentX.BUTTON_LONG_SIDE
        };
    }

    handleMouseDown(e) {
        const mouseX = e.canvasX - this.node.pos[0];
        const mouseY = e.canvasY - this.node.pos[1];

        const hoveredAspect = this.getHoveredAspect(mouseX, mouseY);
        if (hoveredAspect !== null) {
            this.handleAspectClick(hoveredAspect);
            return;
        }

        const hoveredPreset = this.getHoveredPreset(mouseX, mouseY);
        if (hoveredPreset !== null) {
            this.handlePresetClick(hoveredPreset);
        }
    }

    handleWidgetChange(name, value) {
        if (this.activeAspect) {
            this.updateLinkedDimension(name, value);
            this.node.setDirtyCanvas(true);
        }
    }

    drawForeground(ctx) {
        this.drawAspectButtons(ctx);
        this.drawSeparatorLine(ctx);
        this.drawPresetButtons(ctx);
    }

    drawAspectButtons(ctx) {
        let x = EmptyLatentX.BUTTON_MARGIN;
        for (const [ratio, aspect] of Object.entries(this.aspectRatios)) {
            const { width: buttonWidth, height: buttonHeight } =
                this.getButtonDimensions(ratio, aspect);

            const buttonY = EmptyLatentX.BUTTONS_Y +
                (EmptyLatentX.BUTTON_BASE_SIZE - buttonHeight) / 2;

            ctx.fillStyle = this.getButtonColor(ratio === this.activeAspect,
                                              ratio === this.hoveredAspect);

            ctx.beginPath();
            ctx.roundRect(x, buttonY, buttonWidth, buttonHeight, 4);
            ctx.fill();

            this.drawButtonText(ctx, ratio, x + buttonWidth/2,
                              EmptyLatentX.BUTTONS_Y + EmptyLatentX.BUTTON_BASE_SIZE/2);

            x += buttonWidth + EmptyLatentX.BUTTON_MARGIN;
        }
    }

    drawSeparatorLine(ctx) {
        const lineY = EmptyLatentX.BUTTONS_Y + EmptyLatentX.BUTTON_BASE_SIZE + 20;
        ctx.beginPath();
        ctx.strokeStyle = "rgba(100, 100, 100, 0.5)";
        ctx.lineWidth = 1;
        ctx.moveTo(EmptyLatentX.BUTTON_MARGIN, lineY);
        ctx.lineTo(this.node.size[0] - EmptyLatentX.BUTTON_MARGIN, lineY);
        ctx.stroke();
    }

    drawPresetButtons(ctx) {
        const lineY = EmptyLatentX.BUTTONS_Y + EmptyLatentX.BUTTON_BASE_SIZE + 20;
        const presetsStartY = lineY + 20;

        this.presets.forEach((preset, index) => {
            const row = Math.floor(index / EmptyLatentX.PRESETS_PER_ROW);
            const col = index % EmptyLatentX.PRESETS_PER_ROW;

            const presetX = this.presetMargin +
                col * (EmptyLatentX.PRESET_BUTTON_WIDTH + this.presetMargin);
            const presetY = presetsStartY +
                row * (EmptyLatentX.PRESET_BUTTON_HEIGHT + this.presetMargin);

            ctx.fillStyle = this.getButtonColor(false, index === this.hoveredPreset);

            ctx.beginPath();
            ctx.roundRect(presetX, presetY,
                         EmptyLatentX.PRESET_BUTTON_WIDTH,
                         EmptyLatentX.PRESET_BUTTON_HEIGHT, 4);
            ctx.fill();

            this.drawButtonText(ctx, `${preset.width}x${preset.height}`,
                              presetX + EmptyLatentX.PRESET_BUTTON_WIDTH/2,
                              presetY + EmptyLatentX.PRESET_BUTTON_HEIGHT/2);
        });
    }

    getButtonColor(isActive, isHovered) {
        if (isActive) {
            return isHovered ? this.colors.activeHover : this.colors.active;
        }
        return isHovered ? this.colors.hover : this.colors.normal;
    }

    drawButtonText(ctx, text, x, y) {
        ctx.save();
        ctx.fillStyle = "#ffffff";
        ctx.font = "11px Arial";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(text, x, y);
        ctx.restore();
    }
}

app.registerExtension({
    name: "EmptyLatentX",
    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        if (nodeData.name === "EmptyLatentX") {
            nodeType.prototype.computeSize = function() {
                return [306, 315];
            };

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) onNodeCreated.apply(this, []);
                this.EmptyLatentX = new EmptyLatentX(this);
            };
        }
    },
});
