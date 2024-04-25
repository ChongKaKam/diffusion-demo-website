<template>
    <div class="footer">
      <p>
        😘
        Powered by 
        <el-image src="_static/icons/logo.svg" class='logo'></el-image>
        Vue + 
        <el-image src="_static/icons/bxs-flask.svg" class='logo'></el-image>
        Flask
      </p>
    </div>
</template>

<style>
.footer {
  position: fixed;
  left: 0;
  bottom: 0;
  width: 100%;
  background-color: #f8f9fa;
  color: #212529;
  text-align: center;
  padding: 10px 0;
  border-top: 1px solid #d3d5d6;
}
.logo{
  vertical-align: middle;
  height: 16px;
  width: 16px;
}
</style>