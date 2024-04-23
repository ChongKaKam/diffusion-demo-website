<script setup>
import GithubIcon from './components/GitHubIcon.vue'
import UploadFrame from './components/UploadFrame.vue';
import MaskCanvas from './components/MaskCanvas.vue';
import DownloadFrame from './components/DownloadFrame.vue';

import axios from 'axios'
import { onMounted, onUnmounted } from 'vue';

onMounted(() => {
  window.addEventListener('beforeunload', handleBeforeUnload);
});

onUnmounted(() => {
  window.removeEventListener('beforeunload', handleBeforeUnload);
});

async function handleBeforeUnload(event) {
  // 在这里处理 beforeunload 事件
  // event.preventDefault();
  // event.returnValue = '';
  try {
    const response = await axios.post('/clean', '');
  } catch (error) {
    console.error(error);
  }
}
</script>

<template>
  <h1 class="webTitle">Diffusion Model Demo</h1>
  <div class="infoBar">
    <github-icon></github-icon>
    <p style="font-size: 36px;">Gulab</p> 
  </div>
  <div class="imgCols">
    <div class="imgFrame">
      <!-- 123 -->
      <upload-frame></upload-frame>
    </div>
    <div class="imgFrame">
      <mask-canvas></mask-canvas>
    </div>
    <div class="imgFrame">
      <!-- 123 -->
      <download-frame></download-frame>
    </div>
  </div>
</template>

<style scoped>
h1.webTitle{
  font-size: 56px;
  font-family: 'Helvetica Neue';
  text-align: center;
}
div.infoBar{
  display: flex;
  justify-content: center;
  align-items: center;
}
div.imgCols{
   display: flex;
   justify-content: center;
}
div.imgFrame{
  width: 256px;
  /* height: 256px; */
  margin-left: 32px;
  margin-right: 32px;
  margin-top: 8px;
}
</style>
