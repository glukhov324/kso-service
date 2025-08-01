import React from 'react';
import { Container, Typography, Box } from '@mui/material';

function About() {
  return (
    <Container maxWidth="md" sx={{ py: 5 }}>
      <Typography variant="h3" align="center" gutterBottom>
        О проекте
      </Typography>

      <Box sx={{ mt: 3 }}>
        <Typography variant="h6" paragraph>
          В настоящее время большинство розничных сетей («Пятёрочка», «Магнит», «Мария-Ра» и другие)
          активно используют кассы самообслуживания наряду с обычными кассами. Это позволяет сокращать очереди
          и обслуживать больше покупателей за единицу времени.
        </Typography>

        <Typography variant="h6" paragraph>
          Однако существуют случаи, когда покупатель кладёт дорогой товар на весовую платформу кассы самообслуживания,
          но в интерфейсе выбирает название дешёвого весового товара — такой вид мошенничества называется <strong>пересортом</strong>.
          Подобные действия наносят серьёзный ущерб торговым сетям, приводя к значительным финансовым потерям.
        </Typography>

        <Typography variant="h6" paragraph>
          К кассам самообслуживания прикреплены операторы видеоконтроля, которые
          вручную проверяют соответствие между названием весового товара по версии покупателя и изображением с весовой платформы после его добавления.
          Этот процесс является трудоёмким и неэффективным.
        </Typography>

        <Typography variant="h6">
          Проект направлен на <strong>автоматизацию выявления пересортов</strong>.
        </Typography>
      </Box>
    </Container>
  );
}

export default About;