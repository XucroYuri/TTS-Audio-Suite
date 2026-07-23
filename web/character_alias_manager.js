import { app } from "../../scripts/app.js";
import { openCharacterAliasManager } from "./character_alias_manager_ui.js";

app.registerExtension({
    name: "TTS_Audio_Suite.CharacterAliasManager",

    commands: [{
        id: "TTS.AudioSuite.OpenCharacterAliasManager",
        label: "Open Character Alias Manager",
        icon: "pi pi-book",
        function: () => openCharacterAliasManager(),
    }],
});
