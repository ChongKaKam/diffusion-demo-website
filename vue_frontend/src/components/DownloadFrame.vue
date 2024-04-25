<template>
    <div class="col-frame">
        <p class="img-title">Final Image</p> 
        <div class="demo-image" v-loading="message.message">
            <el-image style="width: 256px; height: 256px;" :src="url" fit="contain">
              <template #error>
                <el-image src="_static/icons/photo.png" class='error-img'></el-image>
              </template>
            </el-image>
        </div>
        <el-button class="down-button" size="small" type="info" plain @click="downloadImg">download</el-button>
    </div>
</template>
  
<script setup>
    import {ref, onUnmounted} from 'vue';
    import axios from 'axios';
    const url = ref('');
    const cnt = ref(0);
    const imgID = ref(0)
    const message = defineProps(['message']);
    const fetchImage = async () => {
      try {
        const response = await axios.get('/down-image-url');
        imgID.value = response.data.id;
        url.value = response.data.url + '?' + response.data.id;
        message.message = false
      } catch (error) {
        // console.error(error);
        url.value = 'png?' + imgID.toString();
        message.message = flase
      }
    };
    const intervalId = setInterval(fetchImage, 1000);
    onUnmounted(()=>{
      clearInterval(intervalId);
    })
    fetchImage();
    // TODO: download image
    const downloadImg = () =>{
      const image = new Image();
      image.src = url.value;
      image.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width = image.width;
        canvas.height = image.height;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(image, 0, 0, image.width, image.height);
        const downloadURL = canvas.toDataURL('image/download');
        const a = document.createElement('a');
        a.href = downloadURL;
        a.download = 'final-image.png'
        a.click()
      }
    }
</script>
  
<style scoped>
  .demo-image{
    text-align: center;
    border: 2px solid black;
    width: 260px;
    height: 260px;
    vertical-align: top;
    background-color: white;
  }
  .img-title{
    font-size: 32px;
    text-align: center;
    margin: 12px;
  }
  .col-frame{
    width: 256px;
    align-items: center;
    text-align: center;
  }
  .down-button{
    margin-top: 24px;
    width: 180px;
    height: 40px;
    font-size: 24px;
    color: rgb(58, 58, 58);
  }
  .upload-image-support-text {
    text-align: center;
    margin-top: 0px;
    font: 15px Inter, sans-serif;
    color: #6e6e6e;
  }
  .error-img{
    display: block;
    width: 32px;
    height:32px;
    margin-top: 112px;
    margin-left: 112px;
  }
</style>
  