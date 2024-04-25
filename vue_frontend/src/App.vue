<script setup>
import GithubIcon from './components/GitHubIcon.vue'
import UploadFrame from './components/UploadFrame.vue';
import MaskCanvas from './components/MaskCanvas.vue';
import DownloadFrame from './components/DownloadFrame.vue';
import PageFooter from './components/PageFooter.vue';

import axios from 'axios'
import { onMounted, onUnmounted, ref } from 'vue';

onMounted(() => {
  window.addEventListener('beforeunload', handleBeforeUnload);
});

onUnmounted(() => {
  window.removeEventListener('beforeunload', handleBeforeUnload);
});

async function handleBeforeUnload(event) {
  event.preventDefault();
  console.log('clean images')
  event.returnValue = '';
  try {
    const response = await axios.post('/clean', '');
  } catch (error) {
    console.error(error);
  }
}
let message = ref(false);
const changeMessage = (newMessage) => {
  message.value = newMessage
};
</script>

<template>
  <div class="pageBackground"><img></div>
  <h1 class="webTitle"><u>Diffusion Model Demo</u></h1>
  <div class="infoBar">
    <github-icon></github-icon>
    <p style="font-size: 28px;">Jiawei Zhang, Jiaxin Zhuang</p> 
  </div>
  <div class="colsContainer">
    <div class="imgCols">
      <div class="imgFrame">
        <!-- 123 -->
        <upload-frame></upload-frame>
      </div>
      <div class="imgFrame">
        <mask-canvas :message="message" @changeMessage="changeMessage"></mask-canvas>
      </div>
      <div class="imgFrame">
        <!-- 123 -->
        <download-frame :message="message"></download-frame>
      </div>
    </div>
  </div>
  <page-footer></page-footer>
</template>

<style scoped>
.pageBackground{
  width: 100%;
  height: 100%;
  z-index: -1;
  position: absolute;
}
h1.webTitle{
  margin-top: 16px;
  font-size: 56px;
  font-family: 'Helvetica Neue';
  text-align: center;
}
div.infoBar{
  display: flex;
  justify-content: center;
  align-items: center;
}
div.colsContainer{
  display: flex;
  justify-content: center;
  margin-top: 16px;
}
div.imgCols{
   display: flex;
   justify-content: center;
   width: 1200px;
   padding-bottom: 64px;
   border: 2px dashed #8e8e8e;
   background-color: #F5F7FA;
   border-radius: 30px;
}
div.imgFrame{
  width: 256px;
  /* height: 256px; */
  margin-left: 32px;
  margin-right: 32px;
  margin-top: 8px;
}

</style>
