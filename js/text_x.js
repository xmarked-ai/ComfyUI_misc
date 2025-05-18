// text_x.js с добавлением работы с protected
import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "TextX",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "TextX") {
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                // Вызываем оригинальный обработчик, если он существует
                onExecuted?.apply(this, [message]);

                // Проверяем, что message не null и не undefined
                if (!message) return;

                // Находим виджеты
                const textWidget = this.widgets.find(w => w.name === "text");
                const protectedWidget = this.widgets.find(w => w.name === "protected");

                // Убедимся, что нашли необходимые виджеты
                if (!textWidget || !protectedWidget) return;

                // Проверяем, включена ли защита
                const isProtected = protectedWidget.value;

                // Если защита включена, не обновляем текстовый виджет
                if (isProtected) return;

                // Проверяем наличие текста для обновления
                let newText = null;

                // Проверяем, существует ли message.text как массив
                if (message.text && Array.isArray(message.text) && message.text.length > 0) {
                    newText = message.text[0];
                }
                // Проверяем формат ui.text, как альтернативу
                else if (message.ui && message.ui.text) {
                    if (Array.isArray(message.ui.text) && message.ui.text.length > 0) {
                        newText = message.ui.text[0];
                    } else {
                        newText = message.ui.text;
                    }
                }

                // Если нашли новый текст, обновляем виджет и включаем защиту
                if (newText !== null) {
                    // Заполняем текстовый виджет
                    textWidget.value = newText;
                    if (textWidget.inputEl) {
                        textWidget.inputEl.value = newText;
                    }

                    // Автоматически включаем защиту
                    protectedWidget.value = true;
                    // Обновляем визуальное состояние
                    if (typeof protectedWidget.onSelected === "function") {
                        protectedWidget.onSelected(true);
                    }
                }
            };
        }
    },
});