const chatBox = document.getElementById("chatBox");
const queryInput = document.getElementById("queryInput");
const sendBtn = document.getElementById("sendBtn");
const clearBtn = document.getElementById("clearBtn");
const modelSelect = document.getElementById("modelSelect");
const exampleButtons = document.querySelectorAll(".example-btn");

function scrollToBottom() {
  chatBox.scrollTop = chatBox.scrollHeight;
}

function escapeHtml(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function createMessage(role, text = "") {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${role} bubble-in`;

  const avatar = document.createElement("div");
  avatar.className = `avatar ${role === "assistant" ? "assistant-avatar" : "user-avatar"}`;
  avatar.textContent = role === "assistant" ? "AI" : "You";

  const bubble = document.createElement("div");
  bubble.className = `bubble ${role === "assistant" ? "assistant-bubble" : "user-bubble"}`;
  bubble.textContent = text;

  if (role === "assistant") {
    wrapper.appendChild(avatar);
    wrapper.appendChild(bubble);
  } else {
    wrapper.appendChild(bubble);
    wrapper.appendChild(avatar);
  }

  chatBox.appendChild(wrapper);
  scrollToBottom();
  return bubble;
}

function typeText(element, text, speed = 7) {
  return new Promise((resolve) => {
    let index = 0;
    element.innerHTML = `<span class="typing-cursor">▌</span>`;

    function step() {
      if (index < text.length) {
        const current = escapeHtml(text.slice(0, index + 1)).replace(/\n/g, "<br>");
        element.innerHTML = `${current}<span class="typing-cursor">▌</span>`;
        index++;
        scrollToBottom();
        setTimeout(step, speed);
      } else {
        const finalText = escapeHtml(text).replace(/\n/g, "<br>");
        element.innerHTML = finalText;
        scrollToBottom();
        resolve();
      }
    }

    step();
  });
}

async function sendMessage() {
  const query = queryInput.value.trim();
  const modelKey = modelSelect.value;

  if (!query) return;

  sendBtn.disabled = true;

  createMessage("user", query);
  queryInput.value = "";

  const assistantBubble = createMessage("assistant", "Thinking...");

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
      sendBtn.disabled = false;
      return;
    }

    await typeText(assistantBubble, data.answer, 6);
  } catch (err) {
    assistantBubble.textContent = `Request failed: ${err.message}`;
  }

  sendBtn.disabled = false;
  scrollToBottom();
}

sendBtn.addEventListener("click", sendMessage);

queryInput.addEventListener("keydown", function (e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

clearBtn.addEventListener("click", () => {
  chatBox.innerHTML = `
    <div class="message assistant bubble-in">
      <div class="avatar assistant-avatar">AI</div>
      <div class="bubble assistant-bubble">
        Hello. Ask a medical knowledge question to start the demo.
      </div>
    </div>
  `;
  queryInput.focus();
});

exampleButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    queryInput.value = btn.textContent.trim();
    queryInput.focus();
  });
});

window.addEventListener("load", () => {
  scrollToBottom();
});
