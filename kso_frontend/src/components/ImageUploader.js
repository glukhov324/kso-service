import React, { useState, useRef } from 'react'; 
import { Button, Typography, Select, MenuItem, FormControl, InputLabel, Grid, Card, CardContent } from '@mui/material';

function ImageUploader() {
  const [fileA, setFileA] = useState(null);
  const [fileB, setFileB] = useState(null);
  const [classifier, setClassifier] = useState('ЛУК РЕПЧАТЫЙ');
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const fileInputA = useRef(null);
  const fileInputB = useRef(null);

  const handleFileChangeA = (event) => {
    setFileA(event.target.files[0]);
  };

  const handleFileChangeB = (event) => {
    setFileB(event.target.files[0]);
  };

  const handleClassifierChange = (event) => {
    setClassifier(event.target.value);
  };

  const handleUpload = async () => {
    if (!fileA || !fileB) {
      setError('Пожалуйста, выберите оба файла.');
      return;
    }

    const formData = new FormData();
    formData.append('file_a', fileA);
    formData.append('file_b', fileB);
    formData.append('user_label', classifier);

    try {
      const response = await fetch('http://127.0.0.1:8000/pair_images/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Ошибка сервера');
      }

      const data = await response.json();
      setResult(data);
      setError('');
    } catch (err) {
      setError('Ошибка при загрузке файлов. Проверьте консоль для деталей.');
      console.error(err);
    }
  };



  const handleClear = () => {
    setFileA(null);
    setFileB(null);
    setClassifier('ЛУК РЕПЧАТЫЙ');
    setResult(null);
    setError('');

    if (fileInputA.current) {
      fileInputA.current.value = '';
    }
    if (fileInputB.current) {
      fileInputB.current.value = '';
    }
  };

  return (
    <Card>
      <CardContent>
        <Typography variant="h4" gutterBottom>
          Загрузка изображений и выбор классификатора
        </Typography>

        <Grid item xs={12}>
          <FormControl fullWidth>
            <InputLabel>Классификатор</InputLabel>
            <Select value={classifier} onChange={handleClassifierChange} label="Классификатор">
              <MenuItem value="ЛУК РЕПЧАТЫЙ">ЛУК РЕПЧАТЫЙ</MenuItem>
              <MenuItem value="КАРТОФЕЛЬ">КАРТОФЕЛЬ</MenuItem>
              <MenuItem value="КАПУСТА БЕЛОКОЧАННАЯ">КАПУСТА БЕЛОКОЧАННАЯ</MenuItem>
              <MenuItem value="МИНТАЙ">МИНТАЙ</MenuItem>
              <MenuItem value="СВЕКЛА">СВЕКЛА</MenuItem>
            </Select>
          </FormControl>
        </Grid>

        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <input
              accept=".png"
              style={{ display: 'none' }}
              id="file-a"
              type="file"
              onChange={handleFileChangeA}
              ref={fileInputA} 
            />
            <label htmlFor="file-a">
              <Button variant="contained" component="span">
                Выберите файл A (.png)
              </Button>
            </label>
            {fileA && <Typography>{fileA.name}</Typography>}
          </Grid>

          <Grid item xs={12} md={6}>
            <input
              accept=".png"
              style={{ display: 'none' }}
              id="file-b"
              type="file"
              onChange={handleFileChangeB}
              ref={fileInputB} 
            />
            <label htmlFor="file-b">
              <Button variant="contained" component="span">
                Выберите файл B (.png)
              </Button>
            </label>
            {fileB && <Typography>{fileB.name}</Typography>}
          </Grid>

          <Grid item xs={12}>
            <Button variant="contained" color="primary" onClick={handleUpload}>
              Загрузить
            </Button>
            <Button variant="outlined" color="secondary" onClick={handleClear} style={{ marginLeft: 16 }}>
              Очистить
            </Button>
          </Grid>

          {error && (
            <Grid item xs={12}>
              <Typography color="error">{error}</Typography>
            </Grid>
          )}

          {result && (
            <Grid item xs={12}>
              <Typography variant="h6">Результат:</Typography>
              <img
                src={`data:image/png;base64,${result.mask_base_64}`}
                alt="Результат"
                style={{ maxWidth: '100%', marginTop: 16 }}
              />
              <Typography>
                {result.product_class === 1 ? 'Товар на весах соответствует названию в интерфейсе' : 'Товар на весах не соответствует названию в интерфейсе'}
              </Typography>
            </Grid>
          )}
        </Grid>
      </CardContent>
    </Card>
  );
}

export default ImageUploader;