<template>
    <a class="githubLink" href="https://github.com/ChongKaKam/diffusion-demo-website" target="_blank">
      <img class="githubIcon" :src="iconSrc" />
    </a>
</template>
  
<script>
  export default {
    data() {
      return {
        iconSrc: ''
      }
    },
    mounted() {
      this.iconSrc = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches
        ? '_static/icons/github-fill-dark.png'
        : '_static/icons/github-fill.png';
    }
  }
</script>

<style>
img.githubIcon{
  width: 36px;
  margin-left: 12px;
  margin-right: 12px;
  /* background-color: white; */
  border-radius: 30%;
}
a.githubLink{
  margin-top: 8px;
}
</style>