document.addEventListener("DOMContentLoaded", function () {
    const btn = document.getElementById("voiceBtn");
    const review = document.getElementById("review");

    if (!btn || !review) return;

    let isListening = false;
    let finalTranscript = "";
    let recognition = null;

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

    const updateButton = function (label, listeningState) {
        btn.innerHTML = label;
        btn.classList.toggle("is-listening", listeningState);
    };

    btn.onclick = function () {
        if (!SpeechRecognition) {
            alert("Speech Recognition is not supported in this browser.");
            return;
        }

        if (isListening && recognition) {
            recognition.stop();
            return;
        }

        if (!recognition) {
            recognition = new SpeechRecognition();
            recognition.lang = "en-US";
            recognition.interimResults = true;
            recognition.continuous = false;
            recognition.maxAlternatives = 1;

            recognition.onstart = function () {
                isListening = true;
                updateButton("🎙 Listening...", true);
            };

            recognition.onresult = function (event) {
                let interimTranscript = "";
                for (let i = event.resultIndex; i < event.results.length; i++) {
                    const result = event.results[i];
                    const transcript = result[0].transcript.trim();
                    if (result.isFinal) {
                        finalTranscript = finalTranscript ? finalTranscript + " " + transcript : transcript;
                    } else {
                        interimTranscript = transcript;
                    }
                }
                review.value = finalTranscript + (finalTranscript && interimTranscript ? " " : "") + interimTranscript;
            };

            recognition.onerror = function (event) {
                isListening = false;
                updateButton("🎤 Speak Review", false);
                console.warn("Speech recognition error:", event.error);
            };

            recognition.onend = function () {
                isListening = false;
                updateButton("🎤 Speak Review", false);
            };
        }

        finalTranscript = "";
        review.value = "";
        try {
            recognition.start();
        } catch (e) {
            console.warn("Failed to start recognition:", e);
        }
    };
});