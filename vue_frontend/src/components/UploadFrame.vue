<template>
    <div class="col-frame" >
        <p class="img-title">Uploaded Image</p> 
        <div class="demo-image">
          <el-image style="width: 256px; height: 256px;" :src="url" fit="contain">
            <template #error >
              <el-image src="_static/icons/photo.png" class='error-img'></el-image>
            </template>
          </el-image>
        </div>
        <!-- <el-button class="upload-button" type="info" plain>upload image</el-button> -->
        <el-upload
            action="/upload"
            :limit="99"
            :on-success="addCnt"
            :show-file-list="false">
            <el-button class="upload-button" size="small" type="info" plain>upload image</el-button>
        </el-upload>
        <div slot="tip" class="upload-image-support-text">support image type:<br> *.jpeg, *.jpg, *.png</div>
    </div>
</template>
  
<script setup>
    import {ref, onUnmounted} from 'vue';
    import axios from 'axios';
    const url = ref('');
    const cnt = ref(0);
    const hasError = ref(false);
    const fetchImage = async () => {
      try {
        const response = await axios.get('/up-image-url');
        url.value = response.data.url + '?' + cnt.value.toString();
        hasError.value = false;
      } catch (error) {
        url.value = 'png?' + cnt.value.toString();
        hasError.value = true;
        console.log(error)
      }
    };
    const intervalId = setInterval(fetchImage, 1000);
    onUnmounted(()=>{
      clearInterval(intervalId);
    });
    fetchImage();
    let addCnt = ()=>{
      cnt.value += 1;
    };
</script>
  
<style scoped>
  .demo-image{
    text-align: center;
    border: 2px solid black;
    width: 260px;
    height: 260px;
    vertical-align: top;
  }
  .img-title{
    font-size: 32px;
    text-align: center;
    margin: 12px;
  }
  .col-frame{
    width: 260px;
    align-items: center;
    text-align: center;
  }
  .upload-button{
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
  