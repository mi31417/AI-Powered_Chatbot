const sendBtn = document.getElementById("send-btn");
const input = document.getElementById("user-input");
const chatBody = document.getElementById("chat-body");
const chatToggle = document.getElementById("chat-toggle");
const chatWidget = document.getElementById("chat-widget");
const chatClose = document.getElementById("chat-close");

// Open widget
chatToggle.addEventListener("click", () => {
    chatWidget.classList.remove("hidden");
});

// Close widget
chatClose.addEventListener("click", () => {
    chatWidget.classList.add("hidden");
});


sendBtn.addEventListener("click", sendMessage);
input.addEventListener("keypress", (e) => {
    if (e.key === "Enter") sendMessage();
});
function getTimestamp() {
    const now = new Date();
    return now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function addUserMessage(text) {
    const msg = `
    <div class="chat-message chat-message--user">
    <div class="chat-message__bubble">
        <p>${text}</p>
        <span class="chat-message__time">${getTimestamp()}</span>
    </div>
    </div>`;
    chatBody.insertAdjacentHTML("beforeend", msg);
}

function addBotMessage(text) {
    const msg = `
    <div class="chat-message chat-message--bot">
    <div class="chat-message__avatar">
        <img src="/static/assets/bot.svg">
    </div>
    <div class="chat-message__bubble">
        <p>${text}</p>
        <span class="chat-message__time">${getTimestamp()}</span>
    </div>
</div>`;
    chatBody.insertAdjacentHTML("beforeend", msg);
}
document.querySelectorAll(".qa-btn").forEach(btn => {
    btn.addEventListener("click", () => {
        const link = btn.getAttribute("data-link");
        window.open(link, "_blank");
    });
});

async function sendMessage() {
    const text = input.value.trim();
    if (!text) return;

    addUserMessage(text);
    input.value = "";

    const response = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text })
    });

    const data = await response.json();
    const reply = data.reply || "Sorry, I couldn't generate a response.";

    addBotMessage(reply);

    chatBody.scrollTop = chatBody.scrollHeight;
}
