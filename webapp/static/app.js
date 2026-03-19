const chatBox = document.getElementById("chatBox");
const queryInput = document.getElementById("queryInput");
const sendBtn = document.getElementById("sendBtn");
const modelSelect = document.getElementById("modelSelect");
const exampleButtons = document.querySelectorAll(".example-btn");

function appendMessage(role, text) {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;

  wrapper.appendChild(bubble);
  chatBox.appendChild(wrapper);
  chatBox.scrollTop = chatBox.scrollHeight;

  return bubble;
}

async function sendMessage() {
  const query = queryInput.value.trim();
  const modelKey = modelSelect.value;

  if (!query) return;

  appendMessage("user", query);
  queryInput.value = "";

  const assistantBubble = appendMessage("assistant", "Generating response...");

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model_key: modelKey,
        query: query
      })
    });

    const data = await response.json();

    if (!response.ok) {
      assistantBubble.textContent = `Error: ${data.error || "Unknown error"}`;
      return;
    }

    assistantBubble.textContent = data.answer;
  } catch (err) {
    assistantBubble.textContent = `Request failed: ${err.message}`;
  }

  chatBox.scrollTop = chatBox.scrollHeight;
}

sendBtn.addEventListener("click", sendMessage);

queryInput.addEventListener("keydown", function (e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

exampleButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    queryInput.value = btn.textContent.trim();
    queryInput.focus();
  });
});
