let currentSlide = 0;
const slides = document.querySelectorAll(".slide");
const totalSlides = slides.length;

function updateCounter() {
  const currentElement = document.getElementById("current-slide");
  const totalElement = document.getElementById("total-slides");
  if (currentElement) currentElement.textContent = currentSlide + 1;
  if (totalElement) totalElement.textContent = totalSlides;
}

function showSlide(n) {
  slides.forEach((slide) => slide.classList.remove("active"));

  if (n >= totalSlides) currentSlide = totalSlides - 1;
  else if (n < 0) currentSlide = 0;
  else currentSlide = n;

  slides[currentSlide].classList.add("active");
  updateCounter();
}

function nextSlide() {
  if (currentSlide < totalSlides - 1) {
    currentSlide++;
    showSlide(currentSlide);
  }
}

function prevSlide() {
  if (currentSlide > 0) {
    currentSlide--;
    showSlide(currentSlide);
  }
}

document.addEventListener("DOMContentLoaded", function () {
  updateCounter();

  document.addEventListener("keydown", function (event) {
    switch (event.key) {
      case "ArrowRight":
      case " ":
        nextSlide();
        break;
      case "ArrowLeft":
        prevSlide();
        break;
    }
  });
});
