import { app } from "../../scripts/app.js";
import {
    characterPreviewUrl,
    loadCharacterAliases,
    resetCharacterAliases,
    saveCharacterAliases,
} from "./character_alias_manager_api.js";
import { ensureCharacterAliasManagerStyles } from "./character_alias_manager_styles.js";

const THEME_KEY = "ttsAudioSuite.characterAliasManager.theme";
let activeOverlay = null;
let nextRowId = 1;

const aliasKey = value => String(value || "").trim().toLocaleLowerCase();
const cloneRecord = record => ({
    id: nextRowId++,
    alias: String(record.alias || ""),
    target: String(record.target || ""),
    language: String(record.language || ""),
    source: record.source === "user" ? "user" : "inherited",
    sourceLabel: String(record.source || "inherited"),
    groupId: String(record.groupId || ""),
    group: String(record.group || "Ungrouped"),
    groupNotes: Array.isArray(record.groupNotes) ? record.groupNotes.map(String) : [],
});

function userSnapshot(state) {
    return JSON.stringify(state.userGroups.map(group => ({
        name: group.name.trim(),
        notes: group.notes,
        aliases: state.userRows.filter(row => row.groupId === group.id).map(({ alias, target, language }) => ({
            alias: alias.trim(), target: target.trim(), language: language.trim().toLowerCase(),
        })),
    })));
}

function effectiveRows(state) {
    const userKeys = new Set(state.userRows.map(row => aliasKey(row.alias)).filter(Boolean));
    return [...state.inheritedRows.filter(row => !userKeys.has(aliasKey(row.alias))), ...state.userRows];
}

function validate(state) {
    const errors = [];
    const seen = new Set();
    for (const row of state.userRows) {
        const key = aliasKey(row.alias);
        if (!key || !row.target.trim()) errors.push("Every user alias needs a name and character voice.");
        if (key && seen.has(key)) errors.push(`Duplicate user alias: ${row.alias.trim()}`);
        seen.add(key);
    }
    const groupNames = new Set();
    for (const group of state.userGroups) {
        const key = aliasKey(group.name);
        if (!key) errors.push("Every user group needs a name.");
        if (key && groupNames.has(key)) errors.push(`Duplicate user group: ${group.name.trim()}`);
        groupNames.add(key);
    }
    return [...new Set(errors)];
}

function makeButton(label, className = "") {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `tts-alias-button ${className}`.trim();
    button.textContent = label;
    return button;
}

function setPayload(state, payload) {
    state.characters = Array.isArray(payload.characters) ? payload.characters : [];
    state.characterDetails = new Map(
        Object.entries(payload.characterDetails || {}).map(([name, details]) => [aliasKey(name), details]),
    );
    state.languages = Array.isArray(payload.languages) ? payload.languages : [];
    state.inheritedRows = (payload.inheritedAliases || []).map(cloneRecord);
    state.userGroups = (payload.userGroups || []).map(group => ({
        id: `group-${nextRowId++}`,
        name: String(group.name || "Ungrouped"),
        notes: Array.isArray(group.notes) ? group.notes.map(String) : [],
    }));
    state.userRows = [];
    for (const group of state.userGroups) {
        const payloadGroup = payload.userGroups[state.userGroups.indexOf(group)];
        state.userRows.push(...(payloadGroup.aliases || []).map(record => cloneRecord({
            ...record, source: "user", groupId: group.id, group: group.name,
        })));
    }
    if (!state.userGroups.length) {
        const records = (payload.aliases || []).filter(record => record.source === "user");
        if (records.length) {
            const group = { id: `group-${nextRowId++}`, name: "Ungrouped", notes: [] };
            state.userGroups.push(group);
            state.userRows = records.map(record => cloneRecord({ ...record, groupId: group.id }));
        }
    }
    state.baseline = userSnapshot(state);
    state.userFile = payload.userFile || "";
}

function stopPreview(state) {
    if (state.previewAudio) {
        state.previewAudio.pause();
        state.previewAudio.removeAttribute("src");
        state.previewAudio.load();
    }
    state.previewAudio = null;
    state.previewRowId = null;
}

function togglePreview(state, row, render) {
    if (!aliasKey(row.target)) return;
    state.error = "";
    if (state.previewRowId === row.id) {
        stopPreview(state);
        render();
        return;
    }

    stopPreview(state);
    const audio = new Audio(characterPreviewUrl(row.target));
    state.previewAudio = audio;
    state.previewRowId = row.id;
    audio.addEventListener("ended", () => {
        if (state.previewAudio !== audio) return;
        stopPreview(state);
        render();
    }, { once: true });
    audio.addEventListener("error", () => {
        if (state.previewAudio !== audio) return;
        stopPreview(state);
        state.error = `Could not preview character voice: ${row.target}`;
        render();
    }, { once: true });
    render();
    audio.play().catch(error => {
        if (state.previewAudio !== audio) return;
        stopPreview(state);
        state.error = `Could not preview character voice: ${error.message}`;
        render();
    });
}

function moveUserRow(state, rowId, groupId, beforeRowId = null) {
    const index = state.userRows.findIndex(row => row.id === rowId);
    if (index < 0) return;
    const [row] = state.userRows.splice(index, 1);
    row.groupId = groupId;
    const beforeIndex = beforeRowId == null ? -1 : state.userRows.findIndex(item => item.id === beforeRowId);
    if (beforeIndex >= 0) state.userRows.splice(beforeIndex, 0, row);
    else {
        const groupRows = state.userRows.map((item, itemIndex) => ({ item, itemIndex }))
            .filter(({ item }) => item.groupId === groupId);
        const insertAt = groupRows.length ? groupRows.at(-1).itemIndex + 1 : state.userRows.length;
        state.userRows.splice(insertAt, 0, row);
    }
}

function clearDragFeedback() {
    activeOverlay?.querySelectorAll(".drag-over-before, .drag-over-after, .drag-over-group")
        .forEach(element => element.classList.remove("drag-over-before", "drag-over-after", "drag-over-group"));
}

function makeRowDragImage(element, event) {
    const ghost = element.cloneNode(true);
    const bounds = element.getBoundingClientRect();
    ghost.classList.remove("dragging");
    ghost.classList.add("tts-alias-drag-ghost");
    ghost.style.width = `${bounds.width}px`;
    ghost.style.height = `${bounds.height}px`;
    (activeOverlay?.querySelector(".tts-alias-sheet") || document.body).appendChild(ghost);
    event.dataTransfer.setDragImage(ghost, Math.min(24, event.offsetX), bounds.height / 2);
    requestAnimationFrame(() => ghost.remove());
}

function createSelect(values, value, placeholder) {
    const select = document.createElement("select");
    const options = [...new Set(values)];
    if (value && !options.some(option => option.toLocaleLowerCase() === value.toLocaleLowerCase())) {
        options.unshift(value);
    }
    const empty = document.createElement("option");
    empty.value = "";
    empty.textContent = placeholder;
    select.appendChild(empty);
    for (const optionValue of options) {
        const option = document.createElement("option");
        option.value = optionValue;
        option.textContent = optionValue;
        select.appendChild(option);
    }
    select.value = value;
    return select;
}

function createRow(state, row, render) {
    const element = document.createElement("div");
    const isUser = row.source === "user";
    const missingTarget = row.target && !state.characterKeys.has(aliasKey(row.target));
    element.className = `tts-alias-row ${isUser ? "user" : "inherited"}${missingTarget ? " invalid" : ""}`;
    element.dataset.rowId = String(row.id);

    const grip = document.createElement("span");
    grip.className = `tts-alias-grip${isUser ? " active" : ""}`;
    grip.textContent = isUser ? "⠿" : "";
    if (isUser) {
        grip.draggable = true;
        grip.title = "Drag to reorder or move to another group";
        grip.addEventListener("dragstart", event => {
            event.dataTransfer.effectAllowed = "move";
            event.dataTransfer.setData("text/x-tts-alias-row", String(row.id));
            makeRowDragImage(element, event);
            element.classList.add("dragging");
        });
        grip.addEventListener("dragend", () => {
            element.classList.remove("dragging");
            clearDragFeedback();
        });
        element.addEventListener("dragover", event => {
            if (!event.dataTransfer.types.includes("text/x-tts-alias-row")) return;
            event.preventDefault();
            event.dataTransfer.dropEffect = "move";
            const after = event.clientY > element.getBoundingClientRect().top + element.offsetHeight / 2;
            clearDragFeedback();
            element.classList.add(after ? "drag-over-after" : "drag-over-before");
            element.dataset.dropPosition = after ? "after" : "before";
        });
        element.addEventListener("dragleave", event => {
            if (!element.contains(event.relatedTarget)) {
                element.classList.remove("drag-over-before", "drag-over-after");
            }
        });
        element.addEventListener("drop", event => {
            const draggedId = Number(event.dataTransfer.getData("text/x-tts-alias-row"));
            if (!draggedId || draggedId === row.id) return;
            event.preventDefault();
            const groupRows = state.userRows.filter(item => item.groupId === row.groupId && item.id !== draggedId);
            const rowIndex = groupRows.findIndex(item => item.id === row.id);
            const beforeRowId = element.dataset.dropPosition === "after"
                ? groupRows[rowIndex + 1]?.id ?? null
                : row.id;
            clearDragFeedback();
            moveUserRow(state, draggedId, row.groupId, beforeRowId);
            render();
        });
    }

    const aliasControl = isUser ? document.createElement("input") : document.createElement("span");
    if (isUser) {
        aliasControl.dataset.aliasControl = "alias";
        aliasControl.value = row.alias;
        aliasControl.placeholder = "Character alias";
        aliasControl.addEventListener("input", () => {
            row.alias = aliasControl.value;
            render();
        });
        aliasControl.addEventListener("keydown", event => {
            if (event.key === "Enter") aliasControl.blur();
        });
    } else {
        aliasControl.className = "tts-alias-readonly";
        aliasControl.textContent = row.alias;
        aliasControl.title = row.alias;
    }

    let targetControl;
    if (isUser) {
        targetControl = createSelect(state.characters, row.target, "Select character voice…");
        targetControl.dataset.aliasControl = "target";
        targetControl.title = missingTarget ? "This character voice is not currently available" : row.target;
        targetControl.addEventListener("change", () => {
            row.target = targetControl.value;
            targetControl.blur();
            render();
        });
    } else {
        targetControl = document.createElement("span");
        targetControl.className = "tts-alias-readonly";
        targetControl.textContent = row.target;
        targetControl.title = missingTarget ? `${row.target} (not currently available)` : row.target;
    }
    const voiceCell = document.createElement("div");
    voiceCell.className = "tts-alias-voice-cell";
    const details = state.characterDetails.get(aliasKey(row.target));
    voiceCell.appendChild(targetControl);
    if (row.target && details && !details.hasReferenceText) {
        const transcriptWarning = document.createElement("span");
        transcriptWarning.className = "tts-alias-transcript-warning";
        transcriptWarning.title = "Audio available — reference transcript missing";
        transcriptWarning.setAttribute("role", "img");
        transcriptWarning.setAttribute("aria-label", "Reference transcript missing");
        voiceCell.appendChild(transcriptWarning);
    }
    if (row.target && details?.hasAudio) {
        const preview = document.createElement("button");
        const isPlaying = state.previewRowId === row.id;
        preview.type = "button";
        preview.className = `tts-alias-preview${isPlaying ? " playing" : ""}`;
        preview.title = isPlaying ? `Stop ${row.target} preview` : `Preview ${row.target}`;
        preview.setAttribute("aria-label", preview.title);
        preview.onclick = event => {
            event.stopPropagation();
            togglePreview(state, row, render);
        };
        voiceCell.appendChild(preview);
    }

    let languageControl;
    if (isUser) {
        languageControl = createSelect(state.languages, row.language, "—");
        languageControl.dataset.aliasControl = "language";
        languageControl.addEventListener("change", () => {
            row.language = languageControl.value;
            languageControl.blur();
            render();
        });
    } else {
        languageControl = document.createElement("span");
        languageControl.className = "tts-alias-readonly";
        languageControl.textContent = row.language || "—";
    }

    const source = document.createElement("span");
    source.className = `tts-alias-source ${isUser ? "user" : "inherited"}`;
    source.textContent = isUser ? "USER" : "INHERITED";
    source.title = isUser ? state.userFile : `Loaded from ${row.sourceLabel}`;

    const action = document.createElement("button");
    action.type = "button";
    action.className = `tts-alias-link${isUser ? " tts-alias-remove" : ""}`;
    action.textContent = isUser ? "Remove" : "Override";
    action.onclick = () => {
        if (isUser) state.userRows = state.userRows.filter(candidate => candidate.id !== row.id);
        else {
            let group = state.userGroups[0];
            if (!group) {
                group = { id: `group-${nextRowId++}`, name: "Ungrouped", notes: [] };
                state.userGroups.push(group);
            }
            state.userRows.push(cloneRecord({ ...row, source: "user", groupId: group.id }));
        }
        render();
    };

    element.append(grip, aliasControl, voiceCell, languageControl, source, action);
    return element;
}

export async function openCharacterAliasManager({ onUpdated } = {}) {
    if (activeOverlay) {
        activeOverlay.querySelector(".tts-alias-sheet")?.focus({ preventScroll: true });
        return;
    }

    ensureCharacterAliasManagerStyles();
    const overlay = document.createElement("div");
    overlay.className = "tts-alias-overlay";
    const panel = document.createElement("section");
    panel.className = "tts-alias-sheet";
    panel.dataset.theme = localStorage.getItem(THEME_KEY) === "dark" ? "dark" : "paper";
    panel.tabIndex = -1;
    panel.setAttribute("role", "dialog");
    panel.setAttribute("aria-modal", "true");
    panel.setAttribute("aria-label", "Character Aliases");
    overlay.appendChild(panel);
    document.body.appendChild(overlay);
    activeOverlay = overlay;

    const state = {
        characters: [], characterDetails: new Map(), languages: [], inheritedRows: [],
        userGroups: [], userRows: [], baseline: "[]", previewAudio: null, previewRowId: null,
        userFile: "", filter: "all", search: "", busy: true, error: "", toast: "",
    };

    const isDirty = () => userSnapshot(state) !== state.baseline;
    const close = async (force = false) => {
        if (!force && isDirty()) {
            const discard = await app.extensionManager.dialog.confirm({
                title: "Discard alias changes?",
                message: "Your unsaved character alias changes will be lost.",
            });
            if (!discard) return;
        }
        stopPreview(state);
        overlay.remove();
        activeOverlay = null;
    };

    const render = () => {
        const oldContent = panel.querySelector(".tts-alias-content");
        const activeControl = document.activeElement?.closest?.("[data-alias-control]");
        const activeRow = activeControl?.closest?.("[data-row-id]");
        const activeGroup = activeControl?.closest?.("[data-group-id]");
        const view = {
            scrollTop: oldContent?.scrollTop || 0,
            scrollLeft: oldContent?.scrollLeft || 0,
            rowId: activeRow?.dataset.rowId || "",
            groupId: activeGroup?.dataset.groupId || "",
            control: activeControl?.dataset.aliasControl || "",
            cursor: activeControl instanceof HTMLInputElement ? activeControl.selectionStart : null,
        };
        state.characterKeys = new Set(state.characters.map(aliasKey));
        const rows = effectiveRows(state);
        const userCount = state.userRows.length;
        const inheritedCount = rows.filter(row => row.source !== "user").length;
        const filterCounts = { all: rows.length, user: userCount, inherited: inheritedCount };
        const query = state.search.trim().toLocaleLowerCase();
        const visibleRows = rows.filter(row => {
            const matchesFilter = state.filter === "all" || row.source === state.filter;
            const matchesSearch = !query || `${row.alias} ${row.target} ${row.language}`.toLocaleLowerCase().includes(query);
            return matchesFilter && matchesSearch;
        });
        const errors = validate(state);
        const missingCount = rows.filter(row => row.target && !state.characterKeys.has(aliasKey(row.target))).length;

        panel.replaceChildren();
        const header = document.createElement("header");
        header.className = "tts-alias-header";
        const heading = document.createElement("div");
        const title = document.createElement("h2");
        title.className = "tts-alias-title";
        title.textContent = "Character Aliases";
        const subtitle = document.createElement("div");
        subtitle.className = "tts-alias-subtitle";
        subtitle.textContent = "Friendly names for voices used in [Character] tags";
        heading.append(title, subtitle);
        const headerActions = document.createElement("div");
        headerActions.className = "tts-alias-header-actions";
        const theme = document.createElement("button");
        theme.type = "button";
        theme.className = "tts-alias-icon-button";
        theme.textContent = panel.dataset.theme === "dark" ? "☀" : "☾";
        theme.title = panel.dataset.theme === "dark" ? "Switch to parchment" : "Switch to dark folio";
        theme.onclick = () => {
            panel.dataset.theme = panel.dataset.theme === "dark" ? "paper" : "dark";
            localStorage.setItem(THEME_KEY, panel.dataset.theme);
            render();
        };
        const closeButton = document.createElement("button");
        closeButton.type = "button";
        closeButton.className = "tts-alias-icon-button tts-alias-close";
        closeButton.textContent = "X";
        closeButton.title = "Close";
        closeButton.onclick = () => close();
        headerActions.append(theme, closeButton);
        header.append(heading, headerActions);

        const toolbar = document.createElement("div");
        toolbar.className = "tts-alias-toolbar";
        const searchWrap = document.createElement("label");
        searchWrap.className = "tts-alias-search";
        const search = document.createElement("input");
        search.type = "search";
        search.dataset.aliasControl = "search";
        search.placeholder = "Search aliases or voices…";
        search.value = state.search;
        search.oninput = () => {
            state.search = search.value;
            render();
        };
        searchWrap.appendChild(search);
        const filters = document.createElement("div");
        filters.className = "tts-alias-filter-group";
        for (const [filter, label] of [["all", "All"], ["user", "User"], ["inherited", "Inherited"]]) {
            const button = document.createElement("button");
            button.type = "button";
            button.className = `tts-alias-filter${state.filter === filter ? " active" : ""}`;
            button.textContent = `${label} ${filterCounts[filter]}`;
            button.onclick = () => { state.filter = filter; render(); };
            filters.appendChild(button);
        }
        const add = makeButton("+ Add alias", "tts-alias-add");
        add.disabled = state.busy;
        add.onclick = () => {
            let group = state.userGroups[0];
            if (!group) {
                group = { id: `group-${nextRowId++}`, name: "Ungrouped", notes: [] };
                state.userGroups.push(group);
            }
            const row = cloneRecord({ alias: "", target: "", language: "", source: "user", groupId: group.id });
            state.userRows.push(row);
            state.filter = "all";
            state.search = "";
            render();
            panel.querySelector(`[data-row-id="${row.id}"] input`)?.focus();
        };
        const addGroup = makeButton("+ Group", "tts-alias-add-group");
        addGroup.disabled = state.busy;
        addGroup.onclick = () => {
            const group = { id: `group-${nextRowId++}`, name: "New Group", notes: [] };
            state.userGroups.push(group);
            state.filter = "user";
            state.search = "";
            render();
            const input = panel.querySelector(`[data-group-id="${group.id}"] input`);
            input?.focus();
            input?.select();
        };
        toolbar.append(searchWrap, filters, addGroup, add);

        const content = document.createElement("div");
        content.className = "tts-alias-content";
        if (state.busy) {
            const loading = document.createElement("div");
            loading.className = "tts-alias-loading";
            loading.textContent = "Opening the character registry…";
            content.appendChild(loading);
        } else {
            const table = document.createElement("div");
            table.className = "tts-alias-table";
            const tableHead = document.createElement("div");
            tableHead.className = "tts-alias-table-head";
            for (const label of ["", "Alias", "Character voice", "Language", "Source", ""]) {
                const cell = document.createElement("span");
                cell.textContent = label;
                tableHead.appendChild(cell);
            }
            table.appendChild(tableHead);
            const appendGroup = (group, groupRows, source, sourceLabel = "") => {
                if (!groupRows.length && source !== "user") return;
                const section = document.createElement("section");
                section.className = `tts-alias-group ${source}`;
                section.dataset.groupId = group.id || "";
                const groupHeader = document.createElement("div");
                groupHeader.className = "tts-alias-group-header";
                const ornament = document.createElement("span");
                ornament.className = "tts-alias-group-ornament";
                ornament.textContent = "";
                if (source === "user") {
                    const name = document.createElement("input");
                    name.value = group.name;
                    name.dataset.aliasControl = "group-name";
                    name.placeholder = "Group name";
                    name.oninput = () => { group.name = name.value; render(); };
                    name.onkeydown = event => {
                        if (event.key === "Enter") name.blur();
                    };
                    const moveUp = makeButton("↑", "tts-alias-group-action");
                    const moveDown = makeButton("↓", "tts-alias-group-action");
                    const removeGroup = makeButton("✕", "tts-alias-group-action tts-alias-group-remove");
                    const groupIndex = state.userGroups.indexOf(group);
                    moveUp.title = "Move group up";
                    moveDown.title = "Move group down";
                    removeGroup.title = groupRows.length ? "Move or remove aliases before deleting this group" : "Delete empty group";
                    moveUp.disabled = groupIndex === 0;
                    moveDown.disabled = groupIndex === state.userGroups.length - 1;
                    removeGroup.disabled = groupRows.length > 0;
                    moveUp.onclick = () => {
                        [state.userGroups[groupIndex - 1], state.userGroups[groupIndex]] = [group, state.userGroups[groupIndex - 1]];
                        render();
                    };
                    moveDown.onclick = () => {
                        [state.userGroups[groupIndex], state.userGroups[groupIndex + 1]] = [state.userGroups[groupIndex + 1], group];
                        render();
                    };
                    removeGroup.onclick = () => {
                        state.userGroups = state.userGroups.filter(item => item.id !== group.id);
                        render();
                    };
                    groupHeader.append(ornament, name, moveUp, moveDown, removeGroup);
                    groupHeader.addEventListener("dragover", event => {
                        if (!event.dataTransfer.types.includes("text/x-tts-alias-row")) return;
                        event.preventDefault();
                        event.dataTransfer.dropEffect = "move";
                        clearDragFeedback();
                        groupHeader.classList.add("drag-over-group");
                    });
                    groupHeader.addEventListener("dragleave", event => {
                        if (!groupHeader.contains(event.relatedTarget)) {
                            groupHeader.classList.remove("drag-over-group");
                        }
                    });
                    groupHeader.addEventListener("drop", event => {
                        const draggedId = Number(event.dataTransfer.getData("text/x-tts-alias-row"));
                        if (!draggedId) return;
                        event.preventDefault();
                        clearDragFeedback();
                        moveUserRow(state, draggedId, group.id);
                        render();
                    });
                } else {
                    const name = document.createElement("strong");
                    name.textContent = group.name;
                    const origin = document.createElement("span");
                    origin.className = "tts-alias-group-origin";
                    origin.textContent = sourceLabel;
                    groupHeader.append(ornament, name, origin);
                }
                section.appendChild(groupHeader);
                if (group.notes?.length) {
                    const notes = document.createElement("div");
                    notes.className = "tts-alias-group-notes";
                    notes.textContent = group.notes.map(note => `# ${note}`).join("  ");
                    section.appendChild(notes);
                }
                groupRows.forEach(row => section.appendChild(createRow(state, row, render)));
                table.appendChild(section);
            };

            const inheritedSections = new Map();
            for (const row of visibleRows.filter(row => row.source !== "user")) {
                const key = `${row.sourceLabel}\u0000${row.group}`;
                if (!inheritedSections.has(key)) inheritedSections.set(key, []);
                inheritedSections.get(key).push(row);
            }
            for (const [key, sectionRows] of inheritedSections) {
                const [sourceLabel, name] = key.split("\u0000");
                appendGroup({ name, notes: sectionRows[0]?.groupNotes || [] }, sectionRows, "inherited", sourceLabel);
            }
            for (const group of state.userGroups) {
                const groupRows = visibleRows.filter(row => row.source === "user" && row.groupId === group.id);
                if (state.filter !== "inherited" && (!query || groupRows.length)) appendGroup(group, groupRows, "user");
            }
            if (!visibleRows.length) {
                const empty = document.createElement("div");
                empty.className = "tts-alias-empty";
                empty.textContent = rows.length ? "No aliases match this view." : "No character aliases yet. Add the first one.";
                table.appendChild(empty);
            }
            content.appendChild(table);
        }

        const message = document.createElement("div");
        message.className = `tts-alias-message${state.error || errors.length ? " error" : ""}`;
        if (state.error) message.textContent = state.error;
        else if (errors.length) message.textContent = errors[0];
        else if (missingCount) message.textContent = `${missingCount} alias${missingCount === 1 ? "" : "es"} point to character voices that are not currently available.`;
        else message.textContent = "Inherited aliases stay untouched. Editing one creates a user override.";

        const footer = document.createElement("footer");
        footer.className = "tts-alias-footer";
        const status = document.createElement("div");
        status.className = `tts-alias-status${isDirty() ? " dirty" : ""}`;
        status.textContent = isDirty() ? `${state.userRows.length} user alias${state.userRows.length === 1 ? "" : "es"} · unsaved changes` : `${filterCounts.all} aliases · ${userCount} user overrides`;
        const reset = makeButton("Reset user overrides", "tts-alias-reset");
        reset.disabled = state.busy || (!state.userRows.length && !state.userGroups.length);
        reset.onclick = async () => {
            const confirmed = await app.extensionManager.dialog.confirm({
                title: "Reset user aliases?",
                message: "This removes every user override and reveals the inherited alias map. Shipped and model-folder files will not be changed.",
            });
            if (!confirmed) return;
            state.busy = true;
            render();
            try {
                const payload = await resetCharacterAliases();
                setPayload(state, payload);
                state.toast = "User overrides reset.";
                state.error = "";
                onUpdated?.(payload);
            } catch (error) {
                state.error = error.message;
            } finally {
                state.busy = false;
                render();
            }
        };
        const cancel = makeButton("Cancel");
        cancel.onclick = () => close();
        const save = makeButton("Save changes", "tts-alias-save");
        save.disabled = state.busy || !isDirty() || errors.length > 0;
        save.onclick = async () => {
            state.busy = true;
            state.error = "";
            render();
            try {
                const groups = state.userGroups.map(group => ({
                    name: group.name,
                    notes: group.notes,
                    aliases: state.userRows.filter(row => row.groupId === group.id)
                        .map(({ alias, target, language }) => ({ alias, target, language })),
                }));
                const payload = await saveCharacterAliases(null, groups);
                setPayload(state, payload);
                state.toast = "Character aliases saved.";
                onUpdated?.(payload);
            } catch (error) {
                state.error = error.message;
            } finally {
                state.busy = false;
                render();
            }
        };
        footer.append(status, reset, cancel, save);
        panel.append(header, toolbar, content, message, footer);
        if (state.toast) {
            const toast = document.createElement("div");
            toast.className = "tts-alias-toast";
            toast.textContent = state.toast;
            panel.appendChild(toast);
            const shownToast = state.toast;
            window.setTimeout(() => {
                if (state.toast === shownToast) { state.toast = ""; render(); }
            }, 2200);
        }
        content.scrollTop = view.scrollTop;
        content.scrollLeft = view.scrollLeft;
        const replacement = view.rowId
            ? panel.querySelector(`[data-row-id="${view.rowId}"] [data-alias-control="${view.control}"]`)
            : view.groupId
                ? panel.querySelector(`[data-group-id="${view.groupId}"] [data-alias-control="${view.control}"]`)
                : panel.querySelector(`[data-alias-control="${view.control}"]`);
        replacement?.focus({ preventScroll: true });
        if (replacement instanceof HTMLInputElement && Number.isInteger(view.cursor)) {
            replacement.setSelectionRange(view.cursor, view.cursor);
        }
    };

    overlay.addEventListener("keydown", event => {
        if (event.key === "Escape") close();
    });
    render();
    panel.focus({ preventScroll: true });
    try {
        setPayload(state, await loadCharacterAliases());
    } catch (error) {
        state.error = error.message;
    } finally {
        state.busy = false;
        render();
    }
}
