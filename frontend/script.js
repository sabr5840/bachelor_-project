async function sendMessage() {
    const input = document.getElementById("user-input");
    const chatBox = document.getElementById("chat-box");

    const message = input.value.trim();
    if (!message) return;

    appendMessage(message, "user");
    input.value = "";

    const loadingElement = appendMessage("AI tænker...", "bot");

    try {
        const response = await fetch("http://127.0.0.1:8000/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ message })
        });

        if (!response.ok) {
            throw new Error("Backend returnerede en fejl");
        }

        const data = await response.json();

        loadingElement.remove();
        appendMessage(data.reply, "bot");
    } catch (error) {
        loadingElement.remove();
        appendMessage("Der opstod en fejl ved kontakt til backend eller AI.", "bot");
        console.error(error);
    }
}

function appendMessage(text, sender) {
    const chatBox = document.getElementById("chat-box");
    const messageDiv = document.createElement("div");

    messageDiv.classList.add("message", sender);
    messageDiv.textContent = text;

    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight;

    return messageDiv;
}

document.getElementById("user-input").addEventListener("keydown", function(event) {
    if (event.key === "Enter") {
        sendMessage();
    }
});