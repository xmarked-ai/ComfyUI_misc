// color_correction_x.js - Custom widget for ColorCorrectionX node in ComfyUI
// This widget dynamically shows/hides sliders based on combo settings

import { app } from "../../scripts/app.js";

// Сохраняем оригинальные свойства виджетов
const originalWidgetProps = {};

const CHANNEL_COLORS = {
    "_r": "red",
    "_g": "green",
    "_b": "blue"
};

// Вспомогательная функция для полного скрытия виджета
function hideWidgetForGood(node, widget, suffix = '') {
  // Сохраняем оригинальные свойства виджета, если еще не сохранены
  if (!originalWidgetProps[widget.name]) {
    originalWidgetProps[widget.name] = {
      type: widget.type,
      computeSize: widget.computeSize,
      serializeValue: widget.serializeValue,
      hidden: widget.hidden
    };
  }

  // Скрываем виджет
  widget.computeSize = () => [0, -4];
  widget.type = "converted-widget" + suffix;
  widget.hidden = true;

  return true;
}

// Вспомогательная функция для показа виджета
function showWidget(widget) {
  // Если сохранены оригинальные свойства, восстанавливаем их
  if (originalWidgetProps[widget.name]) {
    const props = originalWidgetProps[widget.name];
    widget.type = props.type;
    widget.computeSize = props.computeSize;
    widget.serializeValue = props.serializeValue;
    widget.hidden = false;
    return true;
  }

  // Если оригинальные свойства не найдены, просто показываем виджет
  widget.hidden = false;
  return false;
}

// Функция добавления callback с сохранением предыдущего
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
    name: "MyNodes.ColorCorrectionX",

    beforeRegisterNodeDef(nodeType, nodeData) {
        // Проверяем оба возможных имени ноды
        if (nodeData?.name !== "Color Correction X" && nodeData?.name !== "ColorCorrectionX") {
            return;
        }

        // Используем chainCallback как в spline_image_mask.js
        chainCallback(nodeType.prototype, "onNodeCreated", function() {
            // Список всех групп параметров, которые имеют combo поведение
            const paramGroups = [
                {
                    combo: "black_point_master",
                    main: "black_point",
                    channels: ["black_point_r", "black_point_g", "black_point_b"]
                },
                {
                    combo: "white_point_master",
                    main: "white_point",
                    channels: ["white_point_r", "white_point_g", "white_point_b"]
                },
                {
                    combo: "gain_master",
                    main: "gain",
                    channels: ["gain_r", "gain_g", "gain_b"]
                },
                {
                    combo: "multiply_master",
                    main: "multiply",
                    channels: ["multiply_r", "multiply_g", "multiply_b"]
                },
                {
                    combo: "offset_master",
                    main: "offset",
                    channels: ["offset_r", "offset_g", "offset_b"]
                },
                {
                    combo: "gamma_master",
                    main: "gamma",
                    channels: ["gamma_r", "gamma_g", "gamma_b"]
                },
                {
                    combo: "brightness_master",
                    main: "brightness",
                    channels: ["brightness_r", "brightness_g", "brightness_b"]
                }
            ];

            // Инициализируем начальное скрытие
            for (const group of paramGroups) {
                const comboWidget = this.widgets.find(w => w.name === group.combo);

                if (!comboWidget) {
                    console.error(`Widget ${group.combo} not found!`);
                    continue;
                }

                const mainWidget = this.widgets.find(w => w.name === group.main);
                const channelWidgets = group.channels.map(name =>
                    this.widgets.find(w => w.name === name)
                );

                if (!mainWidget || channelWidgets.some(w => !w)) {
                    console.error(`Missing widgets for group ${group.combo}`);
                    continue;
                }

                // По умолчанию combo = true, скрываем канальные виджеты
                if (comboWidget.value) {
                    for (const widget of channelWidgets) {
                        hideWidgetForGood(this, widget);
                    }
                }
                // Если combo = false, скрываем основной виджет
                else {
                    hideWidgetForGood(this, mainWidget);
                }
            }

            // Добавляем обработчики событий для переключателей combo
            for (const group of paramGroups) {
                const comboWidget = this.widgets.find(w => w.name === group.combo);
                if (comboWidget) {
                    // Создаем замыкание для сохранения контекста группы
                    const updateVisibility = (value) => {
                        const mainWidget = this.widgets.find(w => w.name === group.main);
                        const channelWidgets = group.channels.map(name =>
                            this.widgets.find(w => w.name === name)
                        );

                        if (!mainWidget || channelWidgets.some(w => !w)) {
                            console.error(`Missing widgets for callback ${group.combo}`);
                            return;
                        }

                        // Если combo = true, показываем основной виджет и скрываем канальные
                        if (value) {
                            showWidget(mainWidget);
                            for (const widget of channelWidgets) {
                                hideWidgetForGood(this, widget);
                            }
                        }
                        // Если combo = false, скрываем основной виджет и показываем канальные
                        else {
                            hideWidgetForGood(this, mainWidget);
                            for (const widget of channelWidgets) {
                                showWidget(widget);
                            }
                        }

                        // Обновляем только высоту ноды
                        const currentSize = this.size;
                        const newHeight = this.computeSize()[1];
                        this.size = [currentSize[0], newHeight];
                        this.setDirtyCanvas(true, true);
                        app.graph.setDirtyCanvas(true, true);
                    };

                    // Сохраняем старый callback, если он есть
                    const oldCallback = comboWidget.callback;

                    // Устанавливаем новый callback
                    comboWidget.callback = function(value) {
                        if (oldCallback) oldCallback.call(this, value);
                        updateVisibility(value);
                    };
                }
            }

            // Добавляем метод для принудительного обновления после callback
            this.onResize = function() {
                this.setDirtyCanvas(true, true);
                app.graph.setDirtyCanvas(true, true);
            };

            // Вставить setTimeout здесь
            setTimeout(() => {
                // Получаем текущий размер
                const currentSize = this.size;
                // Вычисляем новый размер (высоту)
                const newHeight = this.computeSize()[1];
                // Устанавливаем новый размер, сохраняя ширину
                this.size = [currentSize[0], newHeight];
                this.setDirtyCanvas(true, true);
                app.graph.setDirtyCanvas(true, true);
            }, 200);

        });

        // Обеспечиваем правильную синхронизацию при загрузке ноды
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function(info) {
            if (onConfigure) {
                onConfigure.apply(this, arguments);
            }

            // Перенастраиваем виджеты после загрузки схемы
            setTimeout(() => {
                const paramGroups = [
                    {
                        combo: "black_point_master",
                        main: "black_point",
                        channels: ["black_point_r", "black_point_g", "black_point_b"]
                    },
                    {
                        combo: "white_point_master",
                        main: "white_point",
                        channels: ["white_point_r", "white_point_g", "white_point_b"]
                    },
                    {
                        combo: "gain_master",
                        main: "gain",
                        channels: ["gain_r", "gain_g", "gain_b"]
                    },
                    {
                        combo: "multiply_master",
                        main: "multiply",
                        channels: ["multiply_r", "multiply_g", "multiply_b"]
                    },
                    {
                        combo: "offset_master",
                        main: "offset",
                        channels: ["offset_r", "offset_g", "offset_b"]
                    },
                    {
                        combo: "gamma_master",
                        main: "gamma",
                        channels: ["gamma_r", "gamma_g", "gamma_b"]
                    },
                    {
                        combo: "brightness_master",
                        main: "brightness",
                        channels: ["brightness_r", "brightness_g", "brightness_b"]
                    },
                    {
                        combo: "contrast_master",
                        main: "contrast",
                        channels: ["contrast_r", "contrast_g", "contrast_b"]
                    }
                ];

                // Для каждой группы параметров
                for (const group of paramGroups) {
                    const comboWidget = this.widgets.find(w => w.name === group.combo);
                    if (!comboWidget) continue;

                    const mainWidget = this.widgets.find(w => w.name === group.main);
                    const channelWidgets = group.channels.map(name =>
                        this.widgets.find(w => w.name === name)
                    );

                    if (!mainWidget || channelWidgets.some(w => !w)) continue;

                    // Обновляем видимость в зависимости от сохраненного значения combo
                    if (comboWidget.value) {
                        // Если combo = true, показываем основной и скрываем канальные
                        showWidget(mainWidget);
                        for (const widget of channelWidgets) {
                            hideWidgetForGood(this, widget);
                        }
                    } else {
                        // Если combo = false, скрываем основной и показываем канальные
                        hideWidgetForGood(this, mainWidget);
                        for (const widget of channelWidgets) {
                            showWidget(widget);
                        }
                    }
                }

                // Обновляем размер ноды
                const currentSize = this.size;
                const newHeight = this.computeSize()[1];
                this.size = [currentSize[0], newHeight];
                this.setDirtyCanvas(true, true);
                app.graph.setDirtyCanvas(true, true);

            }, 300);
        };
    }
});
