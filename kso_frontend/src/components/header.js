// src/components/Header.js
import React from 'react';
import AppBar from '@mui/material/AppBar';
import Toolbar from '@mui/material/Toolbar';
import Button from '@mui/material/Button';
import Typography from '@mui/material/Typography';

function Header() {
  return (
    <AppBar position="static">
      <Toolbar>
        <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
          KSO Service
        </Typography>

        <Button color="inherit" href="/">
          Главная
        </Button>

        <Button color="inherit" href="/about">
          О проекте
        </Button>
      </Toolbar>
    </AppBar>
  );
}

export default Header;