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

    function patchTextAreaIdentifier() {
        if (typeof getTextAreaIdentifier !== "function") return false;
        if (getTextAreaIdentifier.__aamPatched) return true;

        const original = getTextAreaIdentifier;
        const patched = function (textArea) {
            const root = artistRoot(textArea);
            if (root?.id === "txt2img_anima_artist_chain") return ".txt2img.p.aam";
            if (root?.id === "img2img_anima_artist_chain") return ".img2img.p.aam";
            return original(textArea);
        };

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

    function scheduleSetup() {
        setupArtistAutocomplete();
        setTimeout(setupArtistAutocomplete, 500);
        setTimeout(setupArtistAutocomplete, 1500);
        setTimeout(setupArtistAutocomplete, 3000);
    }

    if (typeof onUiLoaded === "function") {
        onUiLoaded(scheduleSetup);
    } else {
        window.addEventListener("load", scheduleSetup);
    }

    if (typeof onUiUpdate === "function") {
        onUiUpdate(setupArtistAutocomplete);
    }
})();
