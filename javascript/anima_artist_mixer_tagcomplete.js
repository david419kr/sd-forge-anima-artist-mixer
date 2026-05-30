(function () {
    const artistSelectors = [
        "#txt2img_anima_artist_chain textarea",
        "#img2img_anima_artist_chain textarea",
        "#txt2img_anima_artist_chain input[type='text']",
        "#img2img_anima_artist_chain input[type='text']",
    ];

    function artistAreas() {
        return artistSelectors.flatMap((selector) => Array.from(gradioApp().querySelectorAll(selector)));
    }

    function artistRoot(textArea) {
        return textArea?.closest?.("#txt2img_anima_artist_chain, #img2img_anima_artist_chain") || null;
    }

    function isArtistArea(target) {
        return Boolean(target?.matches?.(artistSelectors.join(",")));
    }

    function keyeditOptions() {
        const currentOpts = globalThis.opts || {};
        return {
            precision: Number(currentOpts.keyedit_precision_attention) || 0.1,
            delimiters: currentOpts.keyedit_delimiters || ".,\\/!?%^*;:{}=`~() ",
            whitespace: currentOpts.keyedit_delimiters_whitespace || ["Tab", "Carriage Return", "Line Feed"],
        };
    }

    function updateArtistInput(target) {
        if (typeof updateInput === "function") {
            updateInput(target);
            return;
        }

        target.dispatchEvent(new Event("input", { bubbles: true }));
        target.dispatchEvent(new Event("change", { bubbles: true }));
    }

    function editArtistAttention(event) {
        const target = event.originalTarget || event.composedPath?.()[0] || event.target;
        if (!isArtistArea(target)) return;
        if (!(event.metaKey || event.ctrlKey)) return;

        const isPlus = event.key === "ArrowUp";
        const isMinus = event.key === "ArrowDown";
        if (!isPlus && !isMinus) return;

        let selectionStart = target.selectionStart;
        let selectionEnd = target.selectionEnd;
        let text = target.value;
        const options = keyeditOptions();

        function selectCurrentParenthesisBlock(open, close) {
            if (selectionStart !== selectionEnd) return false;

            const before = text.substring(0, selectionStart);
            const beforeParen = before.lastIndexOf(open);
            if (beforeParen === -1) return false;

            const beforeClosingParen = before.lastIndexOf(close);
            if (beforeClosingParen !== -1 && beforeClosingParen > beforeParen) return false;

            const after = text.substring(selectionStart);
            const afterParen = after.indexOf(close);
            if (afterParen === -1) return false;

            const afterOpeningParen = after.indexOf(open);
            if (afterOpeningParen !== -1 && afterOpeningParen < afterParen) return false;

            const parenContent = text.substring(beforeParen + 1, selectionStart + afterParen);
            if (/.*:-?[\d.]+/s.test(parenContent)) {
                const lastColon = parenContent.lastIndexOf(":");
                selectionStart = beforeParen + 1;
                selectionEnd = selectionStart + lastColon;
            } else {
                selectionStart = beforeParen + 1;
                selectionEnd = selectionStart + parenContent.length;
            }

            target.setSelectionRange(selectionStart, selectionEnd);
            return true;
        }

        function selectCurrentWord() {
            if (selectionStart !== selectionEnd) return false;

            const whitespaceDelimiters = { "Tab": "\t", "Carriage Return": "\r", "Line Feed": "\n" };
            let delimiters = options.delimiters;
            for (const item of options.whitespace) {
                delimiters += whitespaceDelimiters[item] || "";
            }

            while (!delimiters.includes(text[selectionStart - 1]) && selectionStart > 0) {
                selectionStart--;
            }
            while (!delimiters.includes(text[selectionEnd]) && selectionEnd < text.length) {
                selectionEnd++;
            }
            while (text[selectionStart] === " " && selectionStart < selectionEnd) {
                selectionStart++;
            }
            while (text[selectionEnd - 1] === " " && selectionEnd > selectionStart) {
                selectionEnd--;
            }

            target.setSelectionRange(selectionStart, selectionEnd);
            return true;
        }

        if (!selectCurrentParenthesisBlock("<", ">") && !selectCurrentParenthesisBlock("(", ")") && !selectCurrentParenthesisBlock("[", "]")) {
            selectCurrentWord();
        }

        event.preventDefault();
        event.stopPropagation();

        let closeCharacter = ")";
        let delta = options.precision;
        const start = selectionStart > 0 ? text[selectionStart - 1] : "";
        const end = text[selectionEnd];

        if (start === "<") {
            closeCharacter = ">";
            delta = Number((globalThis.opts || {}).keyedit_precision_extra) || delta;
        } else if ((start === "(" && end === ")") || (start === "[" && end === "]")) {
            let numParen = 0;
            while (text[selectionStart - numParen - 1] === start && text[selectionEnd + numParen] === end) {
                numParen++;
            }

            let weight = start === "[" ? (1 / 1.1) ** numParen : 1.1 ** numParen;
            weight = Math.round(weight / options.precision) * options.precision;

            text = text.slice(0, selectionStart - numParen) + "(" + text.slice(selectionStart, selectionEnd) + ":" + weight + ")" + text.slice(selectionEnd + numParen);
            selectionStart -= numParen - 1;
            selectionEnd -= numParen - 1;
        } else if (start !== "(") {
            while (selectionEnd > selectionStart && text[selectionEnd - 1] === " ") {
                selectionEnd--;
            }

            if (selectionStart === selectionEnd) return;

            text = text.slice(0, selectionStart) + "(" + text.slice(selectionStart, selectionEnd) + ":1.0)" + text.slice(selectionEnd);
            selectionStart++;
            selectionEnd++;
        }

        if (text[selectionEnd] !== ":") return;

        const weightLength = text.slice(selectionEnd + 1).indexOf(closeCharacter) + 1;
        let weight = parseFloat(text.slice(selectionEnd + 1, selectionEnd + weightLength));
        if (isNaN(weight)) return;

        weight += isPlus ? delta : -delta;
        weight = parseFloat(weight.toPrecision(12));
        if (Number.isInteger(weight)) weight += ".0";

        if (closeCharacter === ")" && weight === 1) {
            const endParenPos = text.substring(selectionEnd).indexOf(")");
            text = text.slice(0, selectionStart - 1) + text.slice(selectionStart, selectionEnd) + text.slice(selectionEnd + endParenPos + 1);
            selectionStart--;
            selectionEnd--;
        } else {
            text = text.slice(0, selectionEnd + 1) + weight + text.slice(selectionEnd + weightLength);
        }

        target.focus();
        target.value = text;
        target.selectionStart = selectionStart;
        target.selectionEnd = selectionEnd;
        updateArtistInput(target);
    }

    function hasIdentifierPatch(fn, flagName) {
        const seen = new Set();
        const stack = [fn];

        while (stack.length > 0) {
            const current = stack.pop();
            if (typeof current !== "function" || seen.has(current)) continue;
            if (current[flagName]) return true;

            seen.add(current);
            stack.push(current.__aamOriginal, current.__condeltaOriginal);
        }

        return false;
    }

    function inheritIdentifierPatchState(target, source) {
        for (const key of ["__aamPatched", "__aamOriginal", "__condeltaPatched", "__condeltaOriginal"]) {
            if (Object.prototype.hasOwnProperty.call(source, key)) {
                target[key] = source[key];
            }
        }
    }

    function patchTextAreaIdentifier() {
        if (typeof getTextAreaIdentifier !== "function") return false;
        if (hasIdentifierPatch(getTextAreaIdentifier, "__aamPatched")) return true;

        const original = getTextAreaIdentifier;
        const patched = function (textArea) {
            const root = artistRoot(textArea);
            if (root?.id === "txt2img_anima_artist_chain") return ".txt2img.p.aam";
            if (root?.id === "img2img_anima_artist_chain") return ".img2img.p.aam";
            return original(textArea);
        };

        inheritIdentifierPatchState(patched, original);
        patched.__aamPatched = true;
        patched.__aamOriginal = original;

        try {
            getTextAreaIdentifier = patched;
            globalThis.getTextAreaIdentifier = patched;
        } catch (error) {
            console.debug("Anima Artist Mixer: tagcomplete identifier patch failed", error);
            return false;
        }

        return true;
    }

    function setupArtistAutocomplete() {
        if (!patchTextAreaIdentifier()) return;
        if (typeof addAutocompleteToArea !== "function") return;

        for (const area of artistAreas()) {
            if (!area.classList.contains("autocomplete")) {
                addAutocompleteToArea(area);
            }
        }
    }

    function setupArtistKeyedit() {
        for (const area of artistAreas()) {
            if (area.dataset.aamKeyedit === "true") continue;
            area.addEventListener("keydown", editArtistAttention);
            area.dataset.aamKeyedit = "true";
        }
    }

    function setupArtistControls() {
        setupArtistAutocomplete();
        setupArtistKeyedit();
    }

    function scheduleSetup() {
        setupArtistControls();
        setTimeout(setupArtistControls, 500);
        setTimeout(setupArtistControls, 1500);
        setTimeout(setupArtistControls, 3000);
    }

    if (typeof onUiLoaded === "function") {
        onUiLoaded(scheduleSetup);
    } else {
        window.addEventListener("load", scheduleSetup);
    }

    if (typeof onUiUpdate === "function") {
        onUiUpdate(setupArtistControls);
    }
})();
