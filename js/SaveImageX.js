import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "xmtools.SaveImageX",
    async setup() {
        // Регистрация ноды для типа IMAGE
        LiteGraph.slot_types_default_out["IMAGE"] = LiteGraph.slot_types_default_in["IMAGE"] || [];
        LiteGraph.slot_types_default_out["IMAGE"].push("SaveImageX");
    }
});
