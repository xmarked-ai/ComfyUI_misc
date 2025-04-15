
import { app } from "../../scripts/app.js";

function makeUUID() {
  let dt = new Date().getTime();
  const uuid = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = ((dt + Math.random() * 16) % 16) | 0;
    dt = Math.floor(dt / 16);
    return (c === 'x' ? r : (r & 0x3) | 0x8).toString(16);
  });
  return uuid;
}

function hideWidgetForGood(node, widget, suffix = '') {
  widget.origType = widget.type;
  widget.origComputeSize = widget.computeSize;
  widget.origSerializeValue = widget.serializeValue;
  widget.computeSize = () => [0, -4];
  widget.type = "converted-widget" + suffix;
}

function chainCallback(object, property, callback) {
  if (object == undefined) {
    console.error("Tried to add callback to non-existant object");
    return;
  }
  if (property in object) {
    const callback_orig = object[property];
    object[property] = function () {
      const r = callback_orig.apply(this, arguments);
      callback.apply(this, arguments);
      return r;
    };
  } else {
    object[property] = callback;
  }
}

app.registerExtension({
  name: 'MyNodes.SplineImageMask',

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name === 'SplineImageMask') {
      console.log("Registering SplineImageMask extension");

      chainCallback(nodeType.prototype, "onNodeCreated", function () {
        console.log("Node created!");

        this.coordWidget = this.widgets.find(w => w.name === "coordinates");
        this.pointsStoreWidget = this.widgets.find(w => w.name === "points_store");

        hideWidgetForGood(this, this.coordWidget);
        hideWidgetForGood(this, this.pointsStoreWidget);

        this.points = [];
        this.originalImageWidth = 0;
        this.originalImageHeight = 0;
        this.canvasInfo = null;
        this.zoomHandler = null;

        var element = document.createElement("div");
        this.uuid = makeUUID();
        element.id = `spline-image-mask-${this.uuid}`;

        this.splineEditor = this.addDOMWidget(nodeData.name, "SplineImageMaskWidget", element, {
          serialize: false,
          hideOnZoom: false
        });

        this.setSize([512, 550]);

        this.splineEditor.parentEl = document.createElement("div");
        this.splineEditor.parentEl.className = "spline-image-mask";
        this.splineEditor.parentEl.id = `spline-image-mask-${this.uuid}`;
        this.splineEditor.parentEl.style.width = "100%";
        this.splineEditor.parentEl.style.height = "450px";
        this.splineEditor.parentEl.style.overflow = "hidden";
        this.splineEditor.parentEl.style.position = "relative";
        this.splineEditor.parentEl.style.marginBottom = "10px";
        this.splineEditor.parentEl.innerHTML = "<div style='color: white; text-align: center; padding-top: 200px;'>Выберите изображение и нажмите кнопку Load Image</div>";

        element.appendChild(this.splineEditor.parentEl);

        this.loadButton = this.addWidget("button", "Load Image", null, () => {
          this.loadSelectedImage();
        });

        this.clearButton = this.addWidget("button", "Clear Spline", null, () => {
          this.clearSpline();
        });

        this.clearSpline = () => {
          this.points = [];
          this.updateCoordinatesWidget();
          if (this.canvasElement) {
            const ctx = this.canvasElement.getContext('2d');
            ctx.clearRect(0, 0, this.canvasElement.width, this.canvasElement.height);
          }
        };

        this.loadSelectedImage = () => {
          const imageWidget = this.widgets.find(w => w.name === "image");
          if (!imageWidget || !imageWidget.value) return;

          const imageName = imageWidget.value;
          const editorContainer = document.createElement("div");
          editorContainer.style.width = "100%";
          editorContainer.style.height = "100%";
          editorContainer.style.position = "relative";

          const imgElement = document.createElement("img");
          imgElement.style.maxWidth = "100%";
          imgElement.style.maxHeight = "100%";
          imgElement.style.objectFit = "contain";
          imgElement.style.display = "block";
          imgElement.style.margin = "0 auto";

          const canvasElement = document.createElement("canvas");
          canvasElement.style.position = "absolute";
          canvasElement.style.top = "0";
          canvasElement.style.left = "0";
          canvasElement.style.width = "100%";
          canvasElement.style.height = "100%";
          canvasElement.style.pointerEvents = "none";

          const imageUrl = `/view_image/${encodeURIComponent(imageName)}`;

          imgElement.onload = () => {
            this.originalImageWidth = imgElement.naturalWidth;
            this.originalImageHeight = imgElement.naturalHeight;

            this.splineEditor.parentEl.innerHTML = '';
            editorContainer.appendChild(imgElement);
            editorContainer.appendChild(canvasElement);
            this.splineEditor.parentEl.appendChild(editorContainer);

            const imgWidth = Math.min(512, imgElement.naturalWidth);
            this.setSize([Math.max(400, imgWidth), 550]);

            this.imgElement = imgElement;
            this.canvasElement = canvasElement;
            this.updateCanvasSize(true);

            this.setupSplineEvents();
            this.updateSplineDisplay();

            // Добавляем обработчик зума
            if (this.zoomHandler) {
              app.canvas.ds.removeEventListener('zoom', this.zoomHandler);
            }
            this.zoomHandler = () => this.updateCanvasSize();
            app.canvas.ds.addEventListener('zoom', this.zoomHandler);
          };

          imgElement.onerror = () => {
            const backupUrl = `/view?filename=${encodeURIComponent(imageName)}&type=input&subfolder=&format=jpeg`;
            imgElement.src = backupUrl;
          };

          imgElement.src = imageUrl;

          setTimeout(() => {
            this.updateCanvasSize(canvasElement, imgElement);
            if (this.points.length > 0) {
              this.updateSplineDisplay();
            }
          }, 100);
        };

        this.updateCanvasSize = (initial = false) => {
          if (!this.canvasElement || !this.imgElement) return;

          const container = this.splineEditor.parentEl;
          const imgRect = this.imgElement.getBoundingClientRect();
          const containerRect = container.getBoundingClientRect();
          const zoom = app.canvas.ds?.scale || 1;

          // Корректируем размеры с учетом зума
          const displayWidth = imgRect.width / zoom;
          const displayHeight = imgRect.height / zoom;

          const offsetLeft = (containerRect.width - displayWidth * zoom) / (2 * zoom);
          // const offsetTop = (containerRect.height - displayHeight * zoom) / (2 * zoom);
          const offsetTop = 0;

          this.canvasElement.width = displayWidth;
          this.canvasElement.height = displayHeight;

          this.canvasElement.style.width = `${displayWidth}px`;
          this.canvasElement.style.height = `${displayHeight}px`;
          this.canvasElement.style.left = `${offsetLeft}px`;
          this.canvasElement.style.top = `${offsetTop}px`;

          this.canvasInfo = {
            offsetLeft: offsetLeft,
            offsetTop: offsetTop,
            displayWidth: displayWidth,
            displayHeight: displayHeight,
            scaleX: this.originalImageWidth / displayWidth,
            scaleY: this.originalImageHeight / displayHeight,
            zoom: zoom
          };

          if (!initial) {
            this.updateSplineDisplay();
          }
        };

        this.imageToScreenCoords = (x, y) => {
          if (!this.canvasInfo) return { x, y };
          return {
            x: x / this.canvasInfo.scaleX + this.canvasInfo.offsetLeft,
            y: y / this.canvasInfo.scaleY + this.canvasInfo.offsetTop
          };
        };

        this.screenToImageCoords = (x, y) => {
          if (!this.canvasInfo) return { x, y };
          return {
            x: (x - this.canvasInfo.offsetLeft) * this.canvasInfo.scaleX,
            y: (y - this.canvasInfo.offsetTop) * this.canvasInfo.scaleY
          };
        };

        this.setupSplineEvents = () => {
          const container = this.splineEditor.parentEl;

          const getCorrectedCoords = (e) => {
            const containerRect = container.getBoundingClientRect();
            const zoom = app.canvas.ds?.scale || 1;
            return {
              x: (e.clientX - containerRect.left) / zoom,
              y: (e.clientY - containerRect.top) / zoom
            };
          };

          container.addEventListener('click', (e) => {
            if (!this.imgElement || !this.canvasElement) return;

            // Получаем координаты клика относительно окна
            const x = e.clientX;
            const y = e.clientY;

            // Получаем прямоугольник изображения
            const rect = this.imgElement.getBoundingClientRect();

            // Проверяем, что клик был внутри изображения
            if (x >= rect.left && y >= rect.top && x <= rect.right && y <= rect.bottom) {
              // Вычисляем координаты относительно изображения
              const canvasX = x - rect.left;
              const canvasY = y - rect.top;

              // Рассчитываем координаты в исходном изображении
              const scaleX = this.originalImageWidth / rect.width;
              const scaleY = this.originalImageHeight / rect.height;

              const imageX = Math.round(canvasX * scaleX);
              const imageY = Math.round(canvasY * scaleY);

              this.points.push({ x: imageX, y: imageY });
              this.updateCoordinatesWidget();
              this.updateSplineDisplay();
            }
          });

          container.addEventListener('contextmenu', (e) => {
            e.preventDefault();
            if (!this.imgElement || !this.canvasInfo || this.points.length === 0) return;

            const { x, y } = getCorrectedCoords(e);
            const imageCoords = this.screenToImageCoords(x, y);

            let minDist = Infinity;
            let minIndex = -1;
            const threshold = 20 * Math.max(this.canvasInfo.scaleX, this.canvasInfo.scaleY);

            this.points.forEach((point, i) => {
              const dist = Math.sqrt(
                Math.pow(point.x - imageCoords.x, 2) +
                Math.pow(point.y - imageCoords.y, 2)
              );
              if (dist < minDist && dist < threshold) {
                minDist = dist;
                minIndex = i;
              }
            });

            if (minIndex >= 0) {
              this.points.splice(minIndex, 1);
              this.updateCoordinatesWidget();
              this.updateSplineDisplay();
            }
          });
        };

        this.updateCoordinatesWidget = () => {
          if (this.coordWidget && this.pointsStoreWidget) {
            const coordsString = JSON.stringify(this.points);
            this.coordWidget.value = coordsString;
            this.pointsStoreWidget.value = coordsString;
            if (app.graph) app.graph.setDirtyCanvas(true);
          }
        };

        this.updateSplineDisplay = () => {
          if (!this.canvasElement || !this.canvasInfo || !this.points?.length) return;

          const ctx = this.canvasElement.getContext('2d');
          ctx.clearRect(0, 0, this.canvasElement.width, this.canvasElement.height);

          ctx.beginPath();
          const first = this.imageToScreenCoords(this.points[0].x, this.points[0].y);
          ctx.moveTo(first.x - this.canvasInfo.offsetLeft, first.y - this.canvasInfo.offsetTop);

          for (let i = 1; i < this.points.length; i++) {
            const p = this.imageToScreenCoords(this.points[i].x, this.points[i].y);
            ctx.lineTo(p.x - this.canvasInfo.offsetLeft, p.y - this.canvasInfo.offsetTop);
          }

          if (this.points.length >= 3) {
            ctx.lineTo(first.x - this.canvasInfo.offsetLeft, first.y - this.canvasInfo.offsetTop);
          }

          ctx.strokeStyle = '#00FFFF';
          ctx.lineWidth = 2;
          ctx.stroke();

          if (this.points.length >= 3) {
            ctx.fillStyle = 'rgba(0, 255, 255, 0.2)';
            ctx.fill();
          }

          this.points.forEach((point, i) => {
            const p = this.imageToScreenCoords(point.x, point.y);
            ctx.beginPath();
            ctx.arc(
              p.x - this.canvasInfo.offsetLeft,
              p.y - this.canvasInfo.offsetTop,
              5, 0, Math.PI * 2
            );
            ctx.fillStyle = i === 0 ? '#FF0000' : '#FFFF00';
            ctx.fill();
            ctx.strokeStyle = '#000000';
            ctx.lineWidth = 1;
            ctx.stroke();
            ctx.fillStyle = '#FFFFFF';
            ctx.font = '12px Arial';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(
              i + 1,
              p.x - this.canvasInfo.offsetLeft,
              p.y - this.canvasInfo.offsetTop
            );
          });
        };

        chainCallback(this, "onResize", function(size) {
          if (this.canvasElement && this.imgElement) {
            this.updateCanvasSize();
          }
          const buttonHeight = 80;
          const containerHeight = size[1] - buttonHeight - 50;
          if (this.splineEditor?.parentEl) {
            this.splineEditor.parentEl.style.height = `${containerHeight}px`;
          }
        });

        if (this.pointsStoreWidget?.value) {
          try {
            this.points = JSON.parse(this.pointsStoreWidget.value);
          } catch (e) {
            this.points = [];
          }
        }
      });
    }
  }
});
