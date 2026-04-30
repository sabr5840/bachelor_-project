const chatToggle = document.getElementById("chat-toggle");
const chatWidget = document.getElementById("chat-widget");
const chatClose = document.getElementById("chat-close");
const userInput = document.getElementById("user-input");
let chatHistory = [];

if (chatToggle && chatWidget) {
  chatToggle.addEventListener("click", () => {
    chatWidget.classList.toggle("open");
  });
}

if (chatClose && chatWidget) {
  chatClose.addEventListener("click", () => {
    chatWidget.classList.remove("open");
  });
}

async function sendMessage() {
  const input = document.getElementById("user-input");

  const message = input.value.trim();
  if (!message) return;

  appendMessage(message, "user");
  input.value = "";

  chatHistory.push({
    role: "user",
    content: message
  });

  const loadingElement = appendMessage("Genererer svar...", "bot");

  try {
    const response = await fetch("http://127.0.0.1:8000/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        message: message,
        history: chatHistory.slice(-6)
      })
    });

    if (!response.ok) {
      throw new Error("Backend returnerede en fejl");
    }

    const data = await response.json();

    loadingElement.remove();
    appendMessage(data.reply, "bot");

    chatHistory.push({
      role: "assistant",
      content: data.reply
    });

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

if (userInput) {
  userInput.addEventListener("keydown", function (event) {
    if (event.key === "Enter") {
      sendMessage();
    }
  });
}
