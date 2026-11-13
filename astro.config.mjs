import { defineConfig } from 'astro/config';

export default defineConfig({
  site: 'https://dareka-san.github.io',
  base: '/meshshader',
  vite: {
    build: {
      // シェーダー内の改行をエスケープして、出力 JS を 1 行にまとめる。
      minify: 'terser',
    },
  },
});
