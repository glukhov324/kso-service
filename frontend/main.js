async function submitImage() {
    const imageInput = document.getElementById("imageInput").files[0];
    if (!imageInput) {
      alert("Пожалуйста, выберите изображение.");
      return;
    }
  
    // Формируем данные для отправки на сервер
    const formData = new FormData();
    formData.append("file", imageInput);
  
    try {
      // Отправляем запрос POST на FastAPI
      const response = await fetch("http://localhost:8000/pair_images/predict", {
        method: "POST",
        body: formData
      });
  
      // Получаем ответ в формате JSON
      const result = await response.json();
  
      // Раскодируем изображение из base64 и отображаем его
      const processedImage = document.getElementById("processedImage");
      processedImage.src = "data:image/png;base64," + result.processed_image;
      processedImage.style.display = "block"; // Показываем обработанное изображение
  
      // Отображаем уверенность модели
      document.getElementById("confidence").innerText = "Уверенность модели: " + result.confidence;
    } catch (error) {
      console.error("Ошибка при обработке изображения:", error);
      alert("Ошибка при обработке изображения. Проверьте консоль для деталей.");
    }
  }
  