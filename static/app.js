const form = document.querySelector("#search-form");
const queryField = document.querySelector("#query");
const submitButton = document.querySelector("#submit-button");
const statusNode = document.querySelector("#status");
const answerNode = document.querySelector("#answer");
const matchesNode = document.querySelector("#matches");
const matchCountNode = document.querySelector("#match-count");
const matchTemplate = document.querySelector("#match-template");

function setLoadingState(isLoading, message) {
  submitButton.disabled = isLoading;
  submitButton.textContent = isLoading ? "Searching..." : "Find matches";
  statusNode.textContent = message;
}

function renderMatches(matches) {
  matchesNode.innerHTML = "";
  matchCountNode.textContent = `${matches.length} game${matches.length === 1 ? "" : "s"}`;

  if (!matches.length) {
    matchesNode.innerHTML = "<p class='empty'>No matches returned.</p>";
    return;
  }

  for (const match of matches) {
    const fragment = matchTemplate.content.cloneNode(true);
    const image = fragment.querySelector(".match-image");
    const title = fragment.querySelector(".match-title");
    const score = fragment.querySelector(".match-score");
    const description = fragment.querySelector(".match-description");
    const meta = fragment.querySelector(".match-meta");
    const tags = fragment.querySelector(".match-tags");
    const link = fragment.querySelector(".match-link");

    image.src = match.header_image || "";
    image.alt = match.name;
    image.loading = "lazy";
    title.textContent = match.name;
    score.textContent = `score ${match.score}`;
    description.textContent = match.short_description || "No short description available.";

    const genreText = (match.genres || []).slice(0, 3).join(", ") || "Unknown genre";
    const platformText = Object.entries(match.platforms || {})
      .filter(([, enabled]) => enabled)
      .map(([platform]) => platform)
      .join(", ") || "No platform data";
    meta.textContent = `${genreText} | ${match.release_date || "Unknown release date"} | ${platformText}`;

    for (const tag of (match.tags || []).slice(0, 6)) {
      const pill = document.createElement("span");
      pill.className = "tag";
      pill.textContent = tag;
      tags.appendChild(pill);
    }

    link.href = match.store_page;
    matchesNode.appendChild(fragment);
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const query = queryField.value.trim();
  if (!query) {
    setLoadingState(false, "Enter a description first.");
    return;
  }

  setLoadingState(true, "Querying local RAG pipeline...");
  answerNode.textContent = "Thinking...";
  matchesNode.innerHTML = "";
  matchCountNode.textContent = "0 games";

  try {
    const response = await fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.error || "Request failed");
    }

    answerNode.textContent = payload.answer;
    renderMatches(payload.matches || []);
    setLoadingState(false, `Done. Indexed ${payload.meta.indexed_games} games.`);
  } catch (error) {
    answerNode.textContent = error.message;
    renderMatches([]);
    setLoadingState(false, "Search failed.");
  }
});
