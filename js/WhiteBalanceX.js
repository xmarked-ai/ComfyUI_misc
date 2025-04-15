import { app } from "../../scripts/app.js";

class WhiteBalanceX {
    static BUTTON_SIZE = 30;
    static BUTTON_MARGIN = 16;
    static BUTTON_Y = 230;

    constructor(node) {
        this.node = node;
        this.node.size = [350, 280];

        // Find widgets
        this.redWidget = this.node.widgets.find(w => w.name === "color_red");
        this.greenWidget = this.node.widgets.find(w => w.name === "color_green");
        this.blueWidget = this.node.widgets.find(w => w.name === "color_blue");

        this.initializeData();
        this.setupNodeEventHandlers();
    }

    initializeData() {
        this.colors = {
            normal: "rgba(100, 100, 100, 0.8)",
            hover: "rgba(150, 150, 150, 0.9)",
            active: "rgba(0, 150, 200, 0.8)"
        };

        this.isHoveredPicker = false;
        this.isPickingColor = false;
        this.selectedColor = null;
    }

    setupNodeEventHandlers() {
        this.node.onMouseMove = this.handleMouseMove.bind(this);
        this.node.onMouseDown = this.handleMouseDown.bind(this);
        this.node.onDrawForeground = this.drawForeground.bind(this);
    }


    handleMouseMove(e) {
        const mouseX = e.canvasX - this.node.pos[0];
        const mouseY = e.canvasY - this.node.pos[1];

        const prevHoveredPicker = this.isHoveredPicker;
        this.isHoveredPicker = this.isPointInButton(
            mouseX, mouseY,
            WhiteBalanceX.BUTTON_MARGIN,
            WhiteBalanceX.BUTTON_Y,
            WhiteBalanceX.BUTTON_SIZE,
            WhiteBalanceX.BUTTON_SIZE
        );

        if (prevHoveredPicker !== this.isHoveredPicker) {
            this.node.setDirtyCanvas(true);
        }
    }

    handleMouseDown(e) {
        if (!this.isPickingColor) {
            const mouseX = e.canvasX - this.node.pos[0];
            const mouseY = e.canvasY - this.node.pos[1];

            if (this.isPointInButton(
                mouseX, mouseY,
                WhiteBalanceX.BUTTON_MARGIN,
                WhiteBalanceX.BUTTON_Y,
                WhiteBalanceX.BUTTON_SIZE,
                WhiteBalanceX.BUTTON_SIZE
            )) {
                this.startColorPicking();
                e.preventDefault();
                e.stopPropagation();
            }
        }
    }

    startColorPicking() {
        this.isPickingColor = true;

        // ВАЖНОЕ ИЗМЕНЕНИЕ: Вместо стандартного crosshair создаем свой курсор
        this.createCustomCursor();

        // Регистрируем обработчики на window в capture-фазе
        this.documentMouseDownHandler = this.handleDocumentMouseDown.bind(this);
        this.documentKeyDownHandler = this.handleDocumentKeyDown.bind(this);
        window.addEventListener("mousedown", this.documentMouseDownHandler, { capture: true });
        window.addEventListener("keydown", this.documentKeyDownHandler, { capture: true });

        // ДОБАВЛЯЕМ: событие движения мыши для перемещения курсора
        this.documentMouseMoveHandler = this.handleDocumentMouseMove.bind(this);
        window.addEventListener("mousemove", this.documentMouseMoveHandler, { capture: true });

        // Отключаем взаимодействие с canvas и litegraph
        const canvas = document.querySelector(".graph-canvas");
        if (canvas) {
            this.originalPointerEvents = canvas.style.pointerEvents;
            canvas.style.pointerEvents = "none";
        }
        const litegraphContainer = document.querySelector(".litegraph");
        if (litegraphContainer) {
            this.originalLitegraphPointerEvents = litegraphContainer.style.pointerEvents;
            litegraphContainer.style.pointerEvents = "none";
        }

        // Пытаемся отключить обработчики LiteGraph
        if (window.LGraphCanvas) {
            const canvasInstance = app.graph.canvas;
            if (canvasInstance) {
                this.originalProcessMouseDown = canvasInstance.processMouseDown;
                canvasInstance.processMouseDown = () => {};
            }
        }

        this.node.setDirtyCanvas(true);
    }

    // НОВЫЙ МЕТОД: Создаем пользовательский курсор
    createCustomCursor() {
        // Скрываем стандартный курсор
        document.body.style.cursor = "none";

        // Создаем элемент для кружка с крестиком
        this.customCursor = document.createElement("div");
        this.customCursor.style.position = "absolute";
        this.customCursor.style.width = "30px";
        this.customCursor.style.height = "30px";
        this.customCursor.style.borderRadius = "50%";
        this.customCursor.style.border = "2px solid white";
        this.customCursor.style.boxShadow = "0 0 0 1px black";
        this.customCursor.style.pointerEvents = "none";
        this.customCursor.style.zIndex = "10000";
        this.customCursor.style.transform = "translate(-50%, -50%)";

        // Добавляем перекрестие
        const vLine = document.createElement("div");
        vLine.style.position = "absolute";
        vLine.style.width = "1px";
        vLine.style.height = "14px";
        vLine.style.backgroundColor = "white";
        vLine.style.left = "50%";
        vLine.style.top = "50%";
        vLine.style.transform = "translate(-50%, -50%)";
        vLine.style.boxShadow = "0 0 0 1px black";
        vLine.style.pointerEvents = "none";
        this.customCursor.appendChild(vLine);

        const hLine = document.createElement("div");
        hLine.style.position = "absolute";
        hLine.style.width = "14px";
        hLine.style.height = "1px";
        hLine.style.backgroundColor = "white";
        hLine.style.left = "50%";
        hLine.style.top = "50%";
        hLine.style.transform = "translate(-50%, -50%)";
        hLine.style.boxShadow = "0 0 0 1px black";
        hLine.style.pointerEvents = "none";
        this.customCursor.appendChild(hLine);

        document.body.appendChild(this.customCursor);
    }

    // НОВЫЙ МЕТОД: Обработчик движения мыши для перемещения курсора
    handleDocumentMouseMove(e) {
        if (this.customCursor) {
            this.customCursor.style.left = e.clientX + "px";
            this.customCursor.style.top = e.clientY + "px";
        }
    }

    stopColorPicking() {
        this.isPickingColor = false;

        // ВАЖНОЕ ИЗМЕНЕНИЕ: Удаляем кастомный курсор
        this.removeCustomCursor();

        // Удаляем обработчики
        window.removeEventListener("mousedown", this.documentMouseDownHandler, { capture: true });
        window.removeEventListener("keydown", this.documentKeyDownHandler, { capture: true });
        // УДАЛЯЕМ: обработчик движения мыши
        window.removeEventListener("mousemove", this.documentMouseMoveHandler, { capture: true });

        // Восстанавливаем canvas и litegraph
        const canvas = document.querySelector(".graph-canvas");
        if (canvas && this.originalPointerEvents !== undefined) {
            canvas.style.pointerEvents = this.originalPointerEvents;
        }
        const litegraphContainer = document.querySelector(".litegraph");
        if (litegraphContainer && this.originalLitegraphPointerEvents !== undefined) {
            litegraphContainer.style.pointerEvents = this.originalLitegraphPointerEvents;
        }

        // Восстанавливаем обработчики LiteGraph
        if (window.LGraphCanvas && app.graph.canvas) {
            if (this.originalProcessMouseDown) {
                app.graph.canvas.processMouseDown = this.originalProcessMouseDown;
            }
        }

        this.node.setDirtyCanvas(true);
    }

    // НОВЫЙ МЕТОД: Удаляем кастомный курсор
    removeCustomCursor() {
        if (this.customCursor) {
            document.body.removeChild(this.customCursor);
            this.customCursor = null;
        }
        // Восстанавливаем стандартный курсор
        document.body.style.cursor = "";
    }

    handleDocumentMouseDown(e) {
        if (this.customCursor) {
            this.customCursor.style.display = "none";
        }

        if (!this.isPickingColor) return;

        // Используем html2canvas для захвата экрана
        if (typeof html2canvas !== "undefined") {
            html2canvas(document.body, {
                scale: window.devicePixelRatio, // Учитываем масштаб экрана
                useCORS: true, // Пытаемся обойти CORS
            }).then(canvas => {
                const ctx = canvas.getContext("2d");
                const imageData = ctx.getImageData(e.clientX * window.devicePixelRatio, e.clientY * window.devicePixelRatio, 1, 1).data;
                const r = imageData[0] / 255;
                const g = imageData[1] / 255;
                const b = imageData[2] / 255;

                this.updateColorWidgets(r, g, b);
                this.stopColorPicking();
                e.preventDefault();
                e.stopPropagation();
            }).catch(err => {
                console.warn("html2canvas failed:", err);
                this.fallbackToCanvas(e);
            });
        } else {
            this.fallbackToCanvas(e);
        }
    }

    fallbackToCanvas(e) {
        // Пробуем получить цвет из canvas ComfyUI
        const canvas = document.querySelector(".graph-canvas");
        if (canvas) {
            const ctx = canvas.getContext("2d");
            if (ctx) {
                const rect = canvas.getBoundingClientRect();
                const scaleX = canvas.width / rect.width;
                const scaleY = canvas.height / rect.height;
                const x = (e.clientX - rect.left) * scaleX;
                const y = (e.clientY - rect.top) * scaleY;

                try {
                    const imageData = ctx.getImageData(x, y, 1, 1).data;
                    const r = imageData[0] / 255;
                    const g = imageData[1] / 255;
                    const b = imageData[2] / 255;

                    this.updateColorWidgets(r, g, b);
                    this.stopColorPicking();
                    e.preventDefault();
                    e.stopPropagation();
                    return;
                } catch (err) {
                    console.warn("Failed to get canvas pixel:", err);
                }
            }
        }

        // Последний запасной вариант: цвет из DOM
        const sampler = document.createElement("div");
        sampler.style.position = "absolute";
        sampler.style.left = e.clientX + "px";
        sampler.style.top = e.clientY + "px";
        sampler.style.width = "1px";
        sampler.style.height = "1px";
        sampler.style.pointerEvents = "none";
        document.body.appendChild(sampler);

        const computedStyle = window.getComputedStyle(sampler);
        const bgColor = computedStyle.backgroundColor;
        document.body.removeChild(sampler);

        const rgbMatch = bgColor.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*[\d.]+)?\)/);
        if (rgbMatch) {
            const r = parseInt(rgbMatch[1]) / 255;
            const g = parseInt(rgbMatch[2]) / 255;
            const b = parseInt(rgbMatch[3]) / 255;
            this.updateColorWidgets(r, g, b);
        } else {
            console.warn("Could not determine color, using default.");
            this.updateColorWidgets(1, 1, 1); // Белый по умолчанию
        }

        this.stopColorPicking();
        e.preventDefault();
        e.stopPropagation();
    }

    handleDocumentKeyDown(e) {
        if (e.key === "Escape" && this.isPickingColor) {
            this.stopColorPicking();
            e.preventDefault();
            e.stopPropagation();
        }
    }

    updateColorWidgets(r, g, b) {
        if (this.redWidget) {
            this.redWidget.value = r;
            // Не вызываем callback!
        }
        if (this.greenWidget) {
            this.greenWidget.value = g;
            // Не вызываем callback!
        }
        if (this.blueWidget) {
            this.blueWidget.value = b;
            // Не вызываем callback!
        }

        this.selectedColor = { r, g, b };
        // Только перерисовка
        this.node.setDirtyCanvas(true);
    }

    isPointInButton(x, y, buttonX, buttonY, width, height) {
        return x >= buttonX && x <= buttonX + width &&
               y >= buttonY && y <= buttonY + height;
    }

    drawForeground(ctx) {
        this.drawColorPickerButton(ctx);
        if (this.selectedColor) {
            this.drawColorPreview(ctx, this.selectedColor);
        }
    }

    drawColorPickerButton(ctx) {
        ctx.fillStyle = this.isPickingColor
            ? this.colors.active
            : (this.isHoveredPicker ? this.colors.hover : this.colors.normal);

        ctx.beginPath();
        ctx.roundRect(
            WhiteBalanceX.BUTTON_MARGIN,
            WhiteBalanceX.BUTTON_Y,
            WhiteBalanceX.BUTTON_SIZE,
            WhiteBalanceX.BUTTON_SIZE,
            4
        );
        ctx.fill();

        this.drawEyedropperIcon(
            ctx,
            WhiteBalanceX.BUTTON_MARGIN + WhiteBalanceX.BUTTON_SIZE / 2,
            WhiteBalanceX.BUTTON_Y + WhiteBalanceX.BUTTON_SIZE / 2
        );
    }

    drawColorPreview(ctx, color) {
        ctx.save();
        const previewX = WhiteBalanceX.BUTTON_MARGIN + WhiteBalanceX.BUTTON_SIZE + 10;
        const previewY = WhiteBalanceX.BUTTON_Y;
        const previewSize = WhiteBalanceX.BUTTON_SIZE;

        ctx.fillStyle = `rgb(${Math.round(color.r * 255)}, ${Math.round(color.g * 255)}, ${Math.round(color.b * 255)})`;
        ctx.beginPath();
        ctx.roundRect(previewX, previewY, previewSize, previewSize, 4);
        ctx.fill();

        ctx.fillStyle = "#ffffff";
        ctx.font = "12px Arial";
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(
            `R: ${(color.r).toFixed(3)} G: ${(color.g).toFixed(3)} B: ${(color.b).toFixed(3)}`,
            previewX + previewSize + 10,
            previewY + previewSize / 2
        );
        ctx.restore();
    }

    drawEyedropperIcon(ctx, x, y) {
        ctx.save();
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 2;
        ctx.lineCap = "round";

        ctx.beginPath();
        ctx.moveTo(x - 6, y + 6);
        ctx.lineTo(x - 3, y + 3);
        ctx.lineTo(x + 3, y - 3);
        ctx.lineTo(x + 6, y - 6);
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(x - 6, y + 6, 3, 0, Math.PI * 2);
        ctx.stroke();

        ctx.restore();
    }
}

app.registerExtension({
    name: "WhiteBalanceX",
    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        if (nodeData.name === "WhiteBalanceX") {
            nodeType.prototype.computeSize = function () {
                return [350, 280];
            };

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) onNodeCreated.apply(this, []);
                this.WhiteBalanceX = new WhiteBalanceX(this);
            };

            const onRemoved = nodeType.prototype.onRemoved;
            nodeType.prototype.onRemoved = function () {
                if (this.WhiteBalanceX && this.WhiteBalanceX.isPickingColor) {
                    this.WhiteBalanceX.stopColorPicking();
                }
                if (onRemoved) {
                    onRemoved.call(this);
                }
            };
        }
    },
});