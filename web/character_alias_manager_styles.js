const STYLE_ID = "tts-character-alias-manager-styles";

export function ensureCharacterAliasManagerStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
.tts-alias-overlay {
    position: fixed;
    inset: 0;
    z-index: 100000;
    display: grid;
    place-items: center;
    padding: 24px;
    background: rgba(5, 7, 10, .78);
    backdrop-filter: blur(3px);
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.tts-alias-sheet {
    --paper: #efe1ba;
    --paper-deep: #ddc998;
    --paper-field: rgba(255, 250, 229, .58);
    --ink: #332719;
    --muted: #75664f;
    --rule: rgba(100, 72, 38, .38);
    --rule-strong: #8f6a3b;
    --drag-accent: #9b713b;
    --blue: #185f9e;
    --blue-hover: #104d84;
    --green: #356d38;
    --green-bg: rgba(82, 130, 69, .12);
    --stamp: #6e6252;
    --stamp-bg: rgba(91, 82, 67, .09);
    --danger: #9a3e34;
    --shadow: rgba(0, 0, 0, .58);
    position: relative;
    display: flex;
    flex-direction: column;
    width: min(920px, 95vw);
    height: min(820px, 92vh);
    overflow: hidden;
    color: var(--ink);
    background-color: var(--paper);
    background-image:
        radial-gradient(circle at 12% 18%, rgba(127, 91, 39, .08), transparent 28%),
        radial-gradient(circle at 82% 72%, rgba(104, 66, 25, .07), transparent 34%),
        repeating-linear-gradient(4deg, rgba(91, 62, 25, .018) 0 1px, transparent 1px 5px),
        linear-gradient(145deg, rgba(255,255,255,.3), transparent 38%);
    border: 1px solid var(--rule-strong);
    border-radius: 2px;
    box-shadow:
        0 24px 80px var(--shadow),
        inset 0 0 0 4px var(--paper),
        inset 0 0 0 5px var(--rule-strong),
        inset 0 0 0 8px color-mix(in srgb, var(--paper) 82%, transparent),
        inset 0 0 0 9px var(--rule),
        inset 0 0 34px rgba(94, 61, 22, .12);
}

.tts-alias-sheet[data-theme="dark"] {
    --paper: #171612;
    --paper-deep: #211e18;
    --paper-field: rgba(44, 40, 31, .86);
    --ink: #e3d7bc;
    --muted: #a79b84;
    --rule: rgba(185, 160, 114, .25);
    --rule-strong: #79684e;
    --drag-accent: #b39a6c;
    --blue: #4e9ad0;
    --blue-hover: #68afe0;
    --green: #83bd78;
    --green-bg: rgba(91, 146, 82, .14);
    --stamp: #aaa08f;
    --stamp-bg: rgba(184, 172, 147, .08);
    --danger: #dc7a6d;
    --shadow: rgba(0, 0, 0, .82);
    background-image:
        radial-gradient(circle at 15% 20%, rgba(119, 79, 40, .11), transparent 31%),
        radial-gradient(circle at 84% 68%, rgba(77, 91, 105, .1), transparent 36%),
        repeating-linear-gradient(4deg, rgba(231, 213, 178, .012) 0 1px, transparent 1px 5px),
        linear-gradient(145deg, rgba(255,255,255,.025), transparent 42%);
}

.tts-alias-header {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    gap: 18px;
    padding: 28px 34px 18px;
    border-bottom: 1px solid var(--rule-strong);
    box-shadow: 0 3px 0 -2px var(--rule);
}
.tts-alias-title {
    margin: 0;
    font: 600 30px/1.05 Georgia, "Times New Roman", serif;
    letter-spacing: -.02em;
}
.tts-alias-subtitle { margin-top: 6px; color: var(--muted); font: 15px/1.35 Georgia, "Times New Roman", serif; }
.tts-alias-header-actions { display: flex; align-items: flex-start; gap: 10px; }

.tts-alias-icon-button,
.tts-alias-button {
    position: relative;
    color: var(--ink);
    background: var(--paper-field);
    border: 0;
    border-radius: 0;
    clip-path: polygon(5px 0, calc(100% - 5px) 0, 100% 5px, 100% calc(100% - 5px), calc(100% - 5px) 100%, 5px 100%, 0 calc(100% - 5px), 0 5px);
    box-shadow:
        inset 0 0 0 1px var(--rule-strong),
        inset 0 0 0 3px color-mix(in srgb, var(--paper) 78%, transparent),
        inset 0 0 0 4px var(--rule);
    cursor: pointer;
    transition: background .14s ease, color .14s ease, border-color .14s ease, transform .08s ease;
}
.tts-alias-icon-button { width: 34px; height: 34px; padding: 0; font: 700 18px/1 Georgia, "Times New Roman", serif; }
.tts-alias-close {
    width: 34px;
    height: 34px;
    border: 1px solid var(--rule-strong);
    border-radius: 0;
    clip-path: none;
    color: var(--ink);
    background: color-mix(in srgb, var(--paper) 88%, var(--paper-deep));
    box-shadow:
        inset 0 0 0 3px color-mix(in srgb, var(--paper) 88%, transparent),
        inset 0 0 0 4px var(--rule-strong),
        1px 1px 0 color-mix(in srgb, var(--rule-strong) 30%, transparent);
    font: 500 18px/1 Georgia, "Times New Roman", serif;
}
.tts-alias-close::before { display: none; }
.tts-alias-close:hover { color: var(--blue); border-color: var(--blue); background: var(--paper); }
.tts-alias-icon-button:hover,
.tts-alias-button:hover { border-color: var(--blue); color: var(--blue); }
.tts-alias-button:active,
.tts-alias-icon-button:active { transform: translateY(1px); }
.tts-alias-button:disabled { cursor: default; opacity: .46; transform: none; }

.tts-alias-toolbar {
    position: relative;
    display: grid;
    grid-template-columns: minmax(210px, 1fr) auto auto auto;
    gap: 18px;
    align-items: center;
    padding: 16px 34px;
    border-bottom: 1px solid var(--rule);
}
.tts-alias-toolbar::after,
.tts-alias-table-head::after,
.tts-alias-message::after {
    content: "";
    position: absolute;
    right: 28px;
    bottom: -4px;
    width: 7px;
    height: 7px;
    background: var(--paper);
    border: 1px solid var(--rule-strong);
    transform: rotate(45deg);
    box-shadow: inset 0 0 0 2px var(--paper), inset 0 0 0 3px var(--rule-strong);
}
.tts-alias-search { position: relative; }
.tts-alias-search::before {
    content: "";
    position: absolute;
    z-index: 1;
    left: 13px;
    top: 50%;
    width: 10px;
    height: 10px;
    border: 1.5px solid var(--muted);
    border-radius: 50%;
    transform: translateY(-62%);
    pointer-events: none;
}
.tts-alias-search::after {
    content: "";
    position: absolute;
    z-index: 1;
    left: 23px;
    top: calc(50% + 4px);
    width: 6px;
    height: 1.5px;
    background: var(--muted);
    transform: rotate(45deg);
    transform-origin: left center;
    pointer-events: none;
}
.tts-alias-search input { width: 100%; padding-left: 38px; }

.tts-alias-filter-group {
    display: inline-flex;
    overflow: hidden;
    border: 0;
    border-radius: 0;
    clip-path: polygon(5px 0, calc(100% - 5px) 0, 100% 5px, 100% calc(100% - 5px), calc(100% - 5px) 100%, 5px 100%, 0 calc(100% - 5px), 0 5px);
    box-shadow: inset 0 0 0 1px var(--rule-strong), inset 0 0 0 3px var(--paper), inset 0 0 0 4px var(--rule);
}
.tts-alias-filter {
    min-width: 82px;
    padding: 9px 13px;
    color: var(--ink);
    background: rgba(255,255,255,.05);
    border: 0;
    border-right: 1px solid var(--rule);
    cursor: pointer;
    font: 15px Georgia, "Times New Roman", serif;
}
.tts-alias-filter:last-child { border-right: 0; }
.tts-alias-filter.active { color: #f8f1df; background: var(--blue); box-shadow: inset 0 0 0 1px color-mix(in srgb, white 35%, transparent); }

.tts-alias-button { padding: 9px 15px; font: 600 14px/1.2 Georgia, "Times New Roman", serif; }
.tts-alias-add,
.tts-alias-save {
    isolation: isolate;
    color: #f6f1e5;
    background: var(--blue);
    border: 0;
    border-radius: 0;
    clip-path: polygon(5px 0, calc(100% - 5px) 0, 100% 5px, 100% calc(100% - 5px), calc(100% - 5px) 100%, 5px 100%, 0 calc(100% - 5px), 0 5px);
    box-shadow: none;
}
.tts-alias-add::before,
.tts-alias-save::before {
    content: "";
    position: absolute;
    inset: 3px;
    z-index: 0;
    border: 1px solid color-mix(in srgb, #fff 62%, var(--paper-deep));
    clip-path: polygon(3px 0, calc(100% - 3px) 0, 100% 3px, 100% calc(100% - 3px), calc(100% - 3px) 100%, 3px 100%, 0 calc(100% - 3px), 0 3px);
    pointer-events: none;
}
.tts-alias-add:hover,
.tts-alias-save:hover { color: #fff; background: var(--blue-hover); }

.tts-alias-content {
    flex: 1 1 auto;
    min-height: 0;
    overflow: auto;
    padding: 0 24px;
    scrollbar-color: var(--rule-strong) color-mix(in srgb, var(--paper-deep) 44%, transparent);
    scrollbar-width: thin;
}
.tts-alias-content::-webkit-scrollbar { width: 11px; height: 11px; }
.tts-alias-content::-webkit-scrollbar-track {
    background: color-mix(in srgb, var(--paper-deep) 44%, transparent);
    border-left: 1px solid var(--rule);
}
.tts-alias-content::-webkit-scrollbar-thumb {
    min-height: 34px;
    background: var(--rule-strong);
    border: 3px solid var(--paper);
    border-radius: 0;
}
.tts-alias-content::-webkit-scrollbar-thumb:hover { background: var(--blue); }
.tts-alias-content::-webkit-scrollbar-corner { background: var(--paper); }
.tts-alias-content::-webkit-scrollbar-button { display: none; width: 0; height: 0; }
.tts-alias-table { min-width: 760px; }
.tts-alias-table-head,
.tts-alias-row {
    display: grid;
    grid-template-columns: 28px minmax(130px, .8fr) minmax(250px, 1.5fr) 112px 112px 74px;
    gap: 12px;
    align-items: center;
}
.tts-alias-table-head {
    position: sticky;
    top: 0;
    z-index: 2;
    padding: 14px 10px 11px;
    color: var(--ink);
    background: var(--paper);
    border-bottom: 1px solid var(--rule-strong);
    font: 600 14px Georgia, "Times New Roman", serif;
}
.tts-alias-table-head::after { right: 7px; }
.tts-alias-row {
    position: relative;
    min-height: 58px;
    padding: 8px 10px;
    border-bottom: 1px solid var(--rule);
    transition: opacity 120ms ease, background-color 120ms ease, transform 120ms ease;
}
.tts-alias-row.dragging {
    opacity: .3;
    background: color-mix(in srgb, var(--drag-accent) 10%, transparent);
    transform: scale(.995);
}
.tts-alias-row.drag-over-before::before,
.tts-alias-row.drag-over-after::after {
    content: "";
    position: absolute;
    z-index: 3;
    left: 8px;
    right: 8px;
    height: 2px;
    background: var(--drag-accent);
    box-shadow: 0 0 0 1px color-mix(in srgb, var(--paper) 70%, transparent), 0 0 8px color-mix(in srgb, var(--drag-accent) 42%, transparent);
    animation: tts-alias-drop-cue 140ms ease-out;
}
.tts-alias-row.drag-over-before::before { top: -1px; }
.tts-alias-row.drag-over-after::after { bottom: -1px; }
.tts-alias-row.drag-over-before,
.tts-alias-row.drag-over-after { background: color-mix(in srgb, var(--drag-accent) 9%, transparent); }
.tts-alias-drag-ghost {
    position: fixed;
    z-index: 100000;
    top: -10000px;
    left: -10000px;
    box-sizing: border-box;
    color: var(--ink);
    background: var(--paper);
    border: 1px solid var(--drag-accent);
    box-shadow: 0 10px 28px rgba(0, 0, 0, .28);
    opacity: .92;
    transform: rotate(.35deg);
    pointer-events: none;
}
@keyframes tts-alias-drop-cue {
    from { opacity: 0; transform: scaleX(.92); }
    to { opacity: 1; transform: scaleX(1); }
}
.tts-alias-row.inherited { color: var(--muted); }
.tts-alias-row.invalid { background: rgba(154, 62, 52, .08); }
.tts-alias-grip { color: transparent; text-align: center; user-select: none; }
.tts-alias-grip.active { color: var(--muted); cursor: grab; font-size: 18px; }
.tts-alias-grip.active:active { cursor: grabbing; color: var(--drag-accent); }

.tts-alias-group { position: relative; }
.tts-alias-group + .tts-alias-group { margin-top: 8px; }
.tts-alias-group-header {
    display: grid;
    grid-template-columns: 28px minmax(150px, 1fr) auto auto auto;
    gap: 7px;
    align-items: center;
    min-height: 43px;
    padding: 5px 10px;
    color: var(--ink);
    background: color-mix(in srgb, var(--paper-deep) 45%, transparent);
    border-top: 3px double var(--rule-strong);
    border-bottom: 1px solid var(--rule);
    font: 600 15px Georgia, "Times New Roman", serif;
    transition: background-color 120ms ease, box-shadow 120ms ease;
}
.tts-alias-group-header.drag-over-group {
    background: color-mix(in srgb, var(--drag-accent) 12%, var(--paper-deep));
    box-shadow: inset 3px 0 0 var(--drag-accent);
}
.tts-alias-group.inherited .tts-alias-group-header { grid-template-columns: 28px minmax(150px, 1fr) auto; color: var(--muted); }
.tts-alias-group-ornament {
    position: relative;
    width: 22px;
    height: 22px;
    color: var(--rule-strong);
}
.tts-alias-group-ornament::before,
.tts-alias-group-ornament::after {
    content: "";
    position: absolute;
    inset: 4px;
    border: 1px solid currentColor;
    transform: rotate(45deg);
}
.tts-alias-group-ornament::after {
    inset: 8px;
    background: currentColor;
}
.tts-alias-group-origin { font-size: 11px; font-weight: 400; letter-spacing: .04em; opacity: .75; }
.tts-alias-group-header input { height: 32px; font: 600 15px Georgia, "Times New Roman", serif; }
.tts-alias-group-action { min-width: 30px; padding: 4px 7px; }
.tts-alias-group-remove { color: var(--danger); }
.tts-alias-group-notes {
    padding: 7px 46px;
    color: var(--muted);
    background: color-mix(in srgb, var(--paper-deep) 22%, transparent);
    border-bottom: 1px dashed var(--rule);
    font: italic 12px/1.35 Georgia, "Times New Roman", serif;
}
.tts-alias-sheet input,
.tts-alias-sheet select {
    min-width: 0;
    height: 38px;
    padding: 7px 10px;
    color: var(--ink);
    background: var(--paper-field);
    border: 1px solid var(--rule);
    border-radius: 4px;
    outline: none;
    font: 14px/1.2 ui-sans-serif, system-ui, sans-serif;
}
.tts-alias-sheet .tts-alias-search input { padding-left: 42px; }
.tts-alias-sheet input:focus,
.tts-alias-sheet select:focus { border-color: var(--blue); box-shadow: 0 0 0 2px rgba(44, 119, 178, .15); }
.tts-alias-sheet input::placeholder { color: var(--muted); opacity: .9; }
.tts-alias-sheet option { color: var(--ink); background: var(--paper); }
.tts-alias-row.user input,
.tts-alias-row.user select,
.tts-alias-group.user .tts-alias-group-header input {
    color: var(--ink);
    background: transparent;
    border-color: transparent;
    border-radius: 0;
    box-shadow: none;
    font-family: Georgia, "Times New Roman", serif;
    transition: background .12s ease, border-color .12s ease, box-shadow .12s ease;
}
.tts-alias-row.user select {
    appearance: none;
    cursor: pointer;
}
.tts-alias-row.user input:hover,
.tts-alias-row.user select:hover,
.tts-alias-group.user .tts-alias-group-header input:hover {
    border-bottom-color: var(--rule);
}
.tts-alias-row.user input:focus,
.tts-alias-row.user select:focus,
.tts-alias-group.user .tts-alias-group-header input:focus {
    color: var(--ink);
    background: var(--paper-field);
    border: 1px solid var(--blue);
    border-radius: 4px;
    box-shadow: 0 0 0 2px rgba(44, 119, 178, .15);
}
.tts-alias-row.user select:focus { appearance: auto; }
.tts-alias-readonly { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font: 14px Georgia, "Times New Roman", serif; }

.tts-alias-voice-cell {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto auto;
    gap: 7px;
    align-items: center;
    min-width: 0;
}
.tts-alias-voice-cell > input,
.tts-alias-voice-cell > select,
.tts-alias-voice-cell > .tts-alias-readonly { width: 100%; min-width: 0; }
.tts-alias-transcript-warning {
    width: 8px;
    height: 8px;
    flex: 0 0 auto;
    background: #d69722;
    border: 1px solid color-mix(in srgb, #7a4b00 72%, var(--ink));
    box-shadow: 0 0 0 2px color-mix(in srgb, #d69722 18%, transparent);
    transform: rotate(45deg);
}
.tts-alias-sheet[data-theme="dark"] .tts-alias-transcript-warning {
    background: #e7ad43;
    border-color: #f3c76f;
}
.tts-alias-preview {
    position: relative;
    width: 24px;
    height: 24px;
    padding: 0;
    color: var(--blue);
    background: color-mix(in srgb, var(--paper-field) 72%, transparent);
    border: 1px solid color-mix(in srgb, var(--blue) 42%, var(--rule));
    border-radius: 50%;
    opacity: 0;
    cursor: pointer;
    transform: translateX(3px);
    transition: opacity .12s ease, transform .12s ease, background .12s ease;
}
.tts-alias-preview::before {
    content: "";
    position: absolute;
    top: 50%;
    left: 52%;
    width: 0;
    height: 0;
    border-top: 4px solid transparent;
    border-bottom: 4px solid transparent;
    border-left: 6px solid currentColor;
    transform: translate(-42%, -50%);
}
.tts-alias-preview.playing::before {
    width: 7px;
    height: 7px;
    border: 0;
    background: currentColor;
    transform: translate(-50%, -50%);
}
.tts-alias-voice-cell:hover .tts-alias-preview,
.tts-alias-voice-cell:focus-within .tts-alias-preview,
.tts-alias-preview.playing {
    opacity: 1;
    transform: translateX(0);
}
.tts-alias-preview:hover,
.tts-alias-preview:focus-visible {
    color: #fff;
    background: var(--blue);
    outline: none;
}

.tts-alias-source {
    position: relative;
    justify-self: start;
    padding: 5px 9px 4px;
    border: 0;
    border-radius: 0;
    clip-path: polygon(4px 0, calc(100% - 4px) 0, 100% 4px, 100% calc(100% - 4px), calc(100% - 4px) 100%, 4px 100%, 0 calc(100% - 4px), 0 4px);
    font: 700 11px/1 Georgia, "Times New Roman", serif;
    letter-spacing: .06em;
    box-shadow: inset 0 0 0 1px currentColor, inset 0 0 0 3px var(--paper), inset 0 0 0 4px currentColor;
    text-shadow: 0 1px 0 color-mix(in srgb, var(--paper) 75%, transparent);
}
.tts-alias-source.user { color: var(--green); background: var(--green-bg); transform: rotate(-.5deg); }
.tts-alias-source.inherited { color: var(--stamp); background: var(--stamp-bg); transform: rotate(.45deg); }
.tts-alias-link {
    justify-self: start;
    padding: 3px 0;
    color: var(--blue);
    background: transparent;
    border: 0;
    border-bottom: 1px solid currentColor;
    cursor: pointer;
    font: 13px Georgia, "Times New Roman", serif;
}
.tts-alias-remove { color: var(--danger); border-bottom-color: transparent; }

.tts-alias-message {
    position: relative;
    display: flex;
    align-items: center;
    gap: 9px;
    min-height: 54px;
    margin: 0 34px;
    color: var(--muted);
    border-bottom: 1px solid var(--rule);
    font: 14px/1.35 Georgia, "Times New Roman", serif;
}
.tts-alias-message::after { right: 0; }
.tts-alias-message.error { color: var(--danger); }
.tts-alias-message::before { content: "ⓘ"; font-family: sans-serif; }
.tts-alias-message.error::before { content: "⚠"; }
.tts-alias-empty,
.tts-alias-loading { padding: 52px 20px; color: var(--muted); text-align: center; font: 16px Georgia, "Times New Roman", serif; }

.tts-alias-footer {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 17px 34px 22px;
}
.tts-alias-status { margin-right: auto; color: var(--muted); font: 15px Georgia, "Times New Roman", serif; }
.tts-alias-status.dirty { color: #a66c00; }
.tts-alias-sheet[data-theme="dark"] .tts-alias-status.dirty { color: #e1b457; }
.tts-alias-status.dirty::before { content: "●"; margin-right: 9px; font-size: 10px; }
.tts-alias-reset { margin-right: 4px; }

.tts-alias-toast {
    position: absolute;
    right: 24px;
    bottom: 76px;
    z-index: 4;
    max-width: 420px;
    padding: 10px 13px;
    color: #f5f2e9;
    background: rgba(28, 70, 45, .95);
    border: 1px solid #63906d;
    border-radius: 5px;
    box-shadow: 0 8px 24px rgba(0,0,0,.3);
    font-size: 12px;
}

@media (max-width: 820px) {
    .tts-alias-overlay { padding: 10px; }
    .tts-alias-header { padding: 24px 22px 14px; }
    .tts-alias-toolbar { grid-template-columns: 1fr auto auto; gap: 10px; padding: 13px 22px; }
    .tts-alias-search { grid-column: 1 / -1; }
    .tts-alias-content { padding: 0 12px; }
    .tts-alias-message { margin: 0 22px; }
    .tts-alias-footer { flex-wrap: wrap; padding: 14px 22px 18px; }
    .tts-alias-status { flex-basis: 100%; }
}
`;
    document.head.appendChild(style);
}
