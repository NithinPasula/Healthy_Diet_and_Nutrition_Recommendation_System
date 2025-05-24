document.addEventListener("DOMContentLoaded", function () {
  const API_BASE_URL = "http://localhost:5000";
  const FORM_DATA_ENDPOINT = `${API_BASE_URL}/api/form-data`;
  const RECOMMENDATIONS_ENDPOINT = `${API_BASE_URL}/api/get-recommendations`;

  const patientForm = document.getElementById("patientForm");
  const loadingIndicator = document.getElementById("loading");
  const resultsContainer = document.getElementById("results");
  const backToFormButton = document.getElementById("backToForm");
  const historyButton = document.getElementById("viewHistory");
  const historyContainer = document.getElementById("historyContainer");
  const historyList = document.getElementById("historyList");

  let numericRanges = {};

  fetchFormData();

  patientForm.addEventListener("submit", handleFormSubmit);
  backToFormButton.addEventListener("click", function () {
    resultsContainer.style.display = "none";
    historyContainer.style.display = "none";
    document.querySelector(".form-container").style.display = "block";
    window.scrollTo({ top: 0, behavior: "smooth" });
  });

  document.querySelectorAll('input[type="number"]').forEach((input) => {
    input.addEventListener("input", function () {
      validateNumericInput(this);
    });
  });

  async function fetchFormData() {
    try {
      const response = await fetch(FORM_DATA_ENDPOINT);
      if (!response.ok) {
        throw new Error("Failed to fetch form data");
      }
      const data = await response.json();

      populateDropdown("gender", data.gender);
      populateDropdown("chronic_disease", data.chronic_disease);
      populateDropdown("genetic_risk_factor", data.genetic_risk_factor);
      populateDropdown("allergies", data.allergies);
      populateDropdown("alcohol_consumption", data.alcohol_consumption);
      populateDropdown("smoking_habit", data.smoking_habit);
      populateDropdown("dietary_habits", data.dietary_habits);
      populateDropdown("preferred_cuisine", data.preferred_cuisine);
      populateDropdown("food_aversions", data.food_aversions);

      numericRanges = data.numeric_ranges;

      for (const [field, range] of Object.entries(numericRanges)) {
        const input = document.getElementById(field);
        if (input) {
          input.placeholder = `${range[0]}-${range[1]}`;
          input.min = range[0];
          input.max = range[1];
        }
      }

      setDefaultValues();
    } catch (error) {
      console.error("Error fetching form data:", error);
      alert("Failed to load form data. Please refresh the page.");
    }
  }

  function populateDropdown(id, options) {
    const dropdown = document.getElementById(id);
    if (!dropdown) return;

    options.forEach((option) => {
      const optionElement = document.createElement("option");
      optionElement.value = option;
      optionElement.textContent = option;
      dropdown.appendChild(optionElement);
    });
  }

  function setDefaultValues() {
    document.getElementById("age").value = 45;
    document.getElementById("height_cm").value = 175;
    document.getElementById("weight_kg").value = 75;
    document.getElementById("blood_pressure_systolic").value = 120;
    document.getElementById("blood_pressure_diastolic").value = 80;
    document.getElementById("cholesterol_level").value = 180;
    document.getElementById("blood_sugar_level").value = 95;
    document.getElementById("daily_steps").value = 8000;
    document.getElementById("exercise_frequency").value = 3;
    document.getElementById("sleep_hours").value = 7;
    document.getElementById("caloric_intake").value = 2200;
    document.getElementById("protein_intake").value = 90;
    document.getElementById("carbohydrate_intake").value = 240;
    document.getElementById("fat_intake").value = 70;
  }

  function validateNumericInput(input) {
    const fieldName = input.id;
    const errorElement = document.getElementById(
      `${fieldName.replace("_", "-")}-error`
    );

    if (!errorElement) return true;

    const value = parseFloat(input.value);

    errorElement.textContent = "";

    if (numericRanges[fieldName]) {
      const [min, max] = numericRanges[fieldName];

      if (isNaN(value)) {
        errorElement.textContent = "Please enter a valid number";
        return false;
      }

      if (value < min || value > max) {
        errorElement.textContent = `Value must be between ${min} and ${max}`;
        return false;
      }
    }

    return true;
  }

  function validateForm() {
    let isValid = true;
    document.querySelectorAll('input[type="number"]').forEach((input) => {
      if (!validateNumericInput(input)) {
        isValid = false;
      }
    });

    return isValid;
  }

  async function handleFormSubmit(event) {
    event.preventDefault();

    if (!validateForm()) {
      alert("Please correct the errors in the form");
      return;
    }

    historyContainer.style.display = "none";

    loadingIndicator.style.display = "block";
    document.querySelector(".form-container").style.display = "none";

    const formData = {};
    new FormData(patientForm).forEach((value, key) => {
      formData[key] = isNaN(value) ? value : Number(value);
    });

    try {
      const response = await fetch(RECOMMENDATIONS_ENDPOINT, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        throw new Error("Failed to get recommendations");
      }

      const data = await response.json();

      if (data.success) {
        displayResults(data.recommendations);
      } else {
        throw new Error(data.error || "Unknown error occurred");
      }
    } catch (error) {
      console.error("Error getting recommendations:", error);
      alert("Failed to generate recommendations. Please try again.");
      document.querySelector(".form-container").style.display = "block";
    } finally {
      loadingIndicator.style.display = "none";
    }
  }

  function displayResults(recommendations) {
    document.getElementById("mealPlanType").textContent =
      recommendations.mealPlanType;
    document.getElementById("recommendedCalories").textContent =
      recommendations.recommendedCalories;
    document.getElementById("recommendedProtein").textContent =
      recommendations.recommendedProtein;
    document.getElementById("recommendedCarbs").textContent =
      recommendations.recommendedCarbs;
    document.getElementById("recommendedFats").textContent =
      recommendations.recommendedFats;

    const detailedPlan = recommendations.detailedMealPlan;
    document.getElementById("breakfast").textContent = detailedPlan.breakfast;
    document.getElementById("lunch").textContent = detailedPlan.lunch;
    document.getElementById("dinner").textContent = detailedPlan.dinner;

    const snacksList = document.getElementById("snacks");
    snacksList.innerHTML = "";
    detailedPlan.snacks.forEach((snack) => {
      const li = document.createElement("li");
      li.textContent = snack;
      snacksList.appendChild(li);
    });

    resultsContainer.style.display = "block";
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  historyButton.addEventListener("click", async () => {
    if (historyContainer.style.display === "block") {
      historyContainer.style.display = "none";
      historyButton.textContent = "View All Past Recommendations";
      return;
    }

    try {
      const originalText = historyButton.textContent;
      historyButton.textContent = "Loading History...";
      historyButton.disabled = true;

      const response = await fetch(`${API_BASE_URL}/api/user-history`);
      const data = await response.json();

      historyList.innerHTML = "";

      if (data.success) {
        if (data.history.length === 0) {
          historyList.innerHTML =
            "<li class='list-group-item'>No past recommendations found.</li>";
        } else {
          data.history.forEach((item, index) => {
            const li = document.createElement("li");
            li.classList.add("list-group-item");
            const utcDate = new Date(item.timestamp);
            const istOffset = 5.5 * 60 * 60 * 1000;
            const istDate = new Date(utcDate.getTime() + istOffset);

            const formattedTime = istDate.toLocaleString("en-GB", {
              day: "2-digit",
              month: "short",
              year: "numeric",
              hour: "2-digit",
              minute: "2-digit",
              hour12: true,
            });

            li.innerHTML = `
              <div class="d-flex justify-content-between align-items-center">
                <strong>Recommendation #${index + 1}</strong>
                <small class="text-muted">${formattedTime}</small>
              </div>
              <div class="mt-2">
                <p><em>${item.recommendations.mealPlanType}</em></p>
                <p><strong>Calories:</strong> ${
                  item.recommendations.recommendedCalories
                }, 
                  <strong>Protein:</strong> ${
                    item.recommendations.recommendedProtein
                  }g, 
                  <strong>Carbs:</strong> ${
                    item.recommendations.recommendedCarbs
                  }g, 
                  <strong>Fats:</strong> ${
                    item.recommendations.recommendedFats
                  }g</p>
                <div class="meal-section">
                  <p><strong>Breakfast:</strong> ${
                    item.recommendations.detailedMealPlan.breakfast
                  }</p>
                  <p><strong>Lunch:</strong> ${
                    item.recommendations.detailedMealPlan.lunch
                  }</p>
                  <p><strong>Dinner:</strong> ${
                    item.recommendations.detailedMealPlan.dinner
                  }</p>
                  <p><strong>Snacks:</strong></p>
                  <ul class="snack-list">
                    ${item.recommendations.detailedMealPlan.snacks
                      .map((snack) => `<li>${snack}</li>`)
                      .join("")}
                  </ul>
                </div>
              </div>
            `;
            historyList.appendChild(li);
          });
        }

        historyContainer.style.display = "block";
        historyButton.textContent = "Hide Past Recommendations";

        setTimeout(() => {
          window.scrollTo({
            top: historyContainer.offsetTop - 20,
            behavior: "smooth",
          });
        }, 100);
      } else {
        alert("Failed to fetch history: " + data.error);
        historyButton.textContent = originalText;
      }
    } catch (err) {
      console.error("Error fetching history:", err);
      alert("Error loading history");
      historyButton.textContent = "View All Past Recommendations";
    } finally {
      historyButton.disabled = false;
    }
  });
});
