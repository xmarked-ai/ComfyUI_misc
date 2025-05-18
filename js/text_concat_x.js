import { app } from "../../scripts/app.js";

// Функция добавления callback с сохранением предыдущего
function chainCallback(object, property, callback) {
  if (!object) {
    console.error("Tried to add callback to non-existent object");
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
  name: "MyNodes.TextConcatX",

  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name === "TextConcatX") {
      // console.log("Registering textconcat js");

      chainCallback(nodeType.prototype, "onNodeCreated", function () {
        // Список всех текстовых входов
        const textInputs = Array.from({ length: 16 }, (_, i) => `text_${i + 1}`);

        // Хранилище для удалённых входов
        this.removedInputs = this.removedInputs || {};

        // Сохраняем свойства всех входов
        for (let i = 0; i < textInputs.length; i++) {
          const input = this.inputs.find(inp => inp.name === textInputs[i]);
          if (input) {
            this.removedInputs[textInputs[i]] = { name: input.name, type: input.type };
            // console.log(`Saved input ${textInputs[i]}:`, this.removedInputs[textInputs[i]]);
          }
        }

        // Откладываем обработку входов
        setTimeout(() => {
          this.updateInputsState();
        }, 0);
      });

      // Добавляем метод для обновления состояния входов
      nodeType.prototype.updateInputsState = function() {
        const textInputs = Array.from({ length: 16 }, (_, i) => `text_${i + 1}`);

        // Шаг 1: Сначала проверяем, какие входы реально подключены
        const connectedInputs = new Set();

        // Проверяем все входы на наличие соединений через API графа
        const links = app.graph.links || {};
        for (const linkId in links) {
          const link = links[linkId];
          if (link.target_id === this.id) {
            const inputSlot = link.target_slot;
            const input = this.inputs[inputSlot];
            if (input && textInputs.includes(input.name)) {
              connectedInputs.add(input.name);
              // console.log(`[API] Input ${input.name} is connected through graph API`);
            }
          }
        }

        // Также проверяем через input.link
        for (let i = 0; i < textInputs.length; i++) {
          const inputName = textInputs[i];
          const input = this.inputs.find(inp => inp.name === inputName);
          if (input && input.link !== null) {
            connectedInputs.add(inputName);
            // console.log(`[Link] Input ${inputName} is connected through input.link`);
          }
        }

        // Шаг 2: Находим последний подключённый вход
        let lastConnectedIndex = 0;
        for (let i = 0; i < textInputs.length; i++) {
          if (connectedInputs.has(textInputs[i])) {
            lastConnectedIndex = Math.max(lastConnectedIndex, i + 1);
          }
        }

        // console.log(`Last connected index: ${lastConnectedIndex}`);

        // Шаг 3: Восстанавливаем все входы до lastConnectedIndex + 1
        for (let i = 0; i <= lastConnectedIndex && i < textInputs.length; i++) {
          const inputName = textInputs[i];
          const input = this.inputs.find(inp => inp.name === inputName);

          if (!input && this.removedInputs[inputName]) {
            this.addInput(inputName, this.removedInputs[inputName].type);
            // console.log(`[Update] Added input ${inputName}`);
          }
        }

        // Шаг 4: Удаляем все входы после lastConnectedIndex, НО только если они не подключены
        for (let i = lastConnectedIndex + 1; i < textInputs.length; i++) {
          const inputName = textInputs[i];
          const input = this.inputs.find(inp => inp.name === inputName);

          if (input) {
            // Проверяем, что вход действительно не подключен
            if (!connectedInputs.has(inputName)) {
              this.removeInput(this.inputs.indexOf(input));
              // console.log(`[Update] Removed input ${inputName}`);
            }
          }
        }

        // Обновляем размер ноды
        const currentSize = this.size;
        const newHeight = this.computeSize()[1];
        this.size = [currentSize[0], newHeight];
        this.setDirtyCanvas(true, true);
        app.graph.setDirtyCanvas(true, true);
      };

      chainCallback(nodeType.prototype, "onConnectionsChange", function (type, index, connected, link_info) {
        // Обновляем состояние входов после изменения соединений с небольшой задержкой
        setTimeout(() => {
          this.updateInputsState();
        }, 0);
      });

      // Синхронизация при загрузке графа
      chainCallback(nodeType.prototype, "onConfigure", function (info) {
        // Список всех текстовых входов
        const textInputs = Array.from({ length: 16 }, (_, i) => `text_${i + 1}`);

        // Хранилище для удалённых входов
        this.removedInputs = this.removedInputs || {};

        // Сохраняем свойства всех входов
        for (let i = 0; i < textInputs.length; i++) {
          const input = this.inputs.find(inp => inp.name === textInputs[i]);
          if (input && !this.removedInputs[textInputs[i]]) {
            this.removedInputs[textInputs[i]] = { name: input.name, type: input.type };
          }
        }

        // Откладываем обновление состояния входов
        setTimeout(() => {
          this.updateInputsState();
        }, 0);
      });
    }
  }
});