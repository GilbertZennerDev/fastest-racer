const navActions = document.getElementById("navActions");
const navLogin = document.getElementById("navLogin");
const upgradeBtn = document.getElementById("upgradeBtn");
const authBackdrop = document.getElementById("authBackdrop");
const authClose = document.getElementById("authClose");
const authForm = document.getElementById("authForm");
const authError = document.getElementById("authError");
const authSubmit = document.getElementById("authSubmit");
const tabBtns = document.querySelectorAll(".tab-btn");

let authMode = "login";
let currentUser = null; // { logged_in, email, tier }

function openAuthModal(mode) {
  authMode = mode;
  tabBtns.forEach((b) => b.classList.toggle("active", b.dataset.tab === mode));
  authSubmit.textContent = mode === "login" ? "Log in" : "Sign up";
  authError.hidden = true;
  authBackdrop.hidden = false;
}

function closeAuthModal() {
  authBackdrop.hidden = true;
}

tabBtns.forEach((btn) => btn.addEventListener("click", () => openAuthModal(btn.dataset.tab)));
authClose.addEventListener("click", closeAuthModal);
authBackdrop.addEventListener("click", (e) => { if (e.target === authBackdrop) closeAuthModal(); });

navLogin.addEventListener("click", (e) => {
  e.preventDefault();
  openAuthModal("login");
});

authForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  authError.hidden = true;
  const email = document.getElementById("authEmail").value;
  const password = document.getElementById("authPassword").value;
  const endpoint = authMode === "login" ? "/api/auth/login" : "/api/auth/signup";

  try {
    const res = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Something went wrong.");
    closeAuthModal();
    await refreshAuthState();
  } catch (err) {
    authError.textContent = err.message;
    authError.hidden = false;
  }
});

async function refreshAuthState() {
  const res = await fetch("/api/auth/me");
  currentUser = await res.json();
  renderNav();
}

function renderNav() {
  if (currentUser && currentUser.logged_in) {
    navActions.innerHTML = `
      <span class="btn-ghost" style="cursor:default;">${currentUser.email} · ${currentUser.tier === "pro" ? "Pro" : "Free"}</span>
      <a href="/app/" class="btn-primary">Open app</a>
    `;
  } else {
    navActions.innerHTML = `
      <a href="#" id="navLogin" class="btn-ghost">Log in</a>
      <a href="/app/" id="navLaunch" class="btn-primary">Try it free</a>
    `;
    document.getElementById("navLogin").addEventListener("click", (e) => {
      e.preventDefault();
      openAuthModal("login");
    });
  }
}

upgradeBtn.addEventListener("click", async (e) => {
  e.preventDefault();
  if (!currentUser || !currentUser.logged_in) {
    openAuthModal("signup");
    return;
  }
  try {
    const res = await fetch("/api/billing/checkout", { method: "POST" });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Checkout is not available right now.");
    window.location.href = data.url;
  } catch (err) {
    alert(err.message);
  }
});

refreshAuthState();
