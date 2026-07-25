import { api } from "/scripts/api.js";

const ENDPOINT = "/api/tts-audio-suite/character-aliases";

async function request(path = "", options = {}) {
    const response = await api.fetchApi(`${ENDPOINT}${path}`, options);
    let payload = {};
    try {
        payload = await response.json();
    } catch {
        // The status below is more useful than a secondary JSON parse error.
    }
    if (!response.ok) {
        throw new Error(payload.error || `Character alias request failed (${response.status})`);
    }
    return payload;
}

export function loadCharacterAliases(forceRefresh = false) {
    return request(forceRefresh ? "?refresh=1" : "");
}

export function saveCharacterAliases(aliases, groups) {
    return request("", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ aliases, groups }),
    });
}

export function resetCharacterAliases() {
    return request("/reset", { method: "POST" });
}

export function characterPreviewUrl(characterName) {
    const params = new URLSearchParams({ character_name: characterName });
    return api.apiURL(`/api/tts-audio-suite/character-preview?${params.toString()}`);
}
