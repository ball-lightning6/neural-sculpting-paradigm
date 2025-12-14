// 在主页面的main.js中添加调试信息
// 在浏览器控制台中运行这些代码来调试

// 调试函数
function debugRender() {
    console.log('=== 开始调试渲染 ===');
    console.log('当前语言:', currentLang);
    console.log('当前顶级分类:', currentTopCategory);
    console.log('当前子分类索引:', currentSubCategoryIndex);

    if (currentTopCategory === 'independentProjects') {
        console.log('正在处理独立项目...');
        const category = docsData[currentTopCategory].categories[currentSubCategoryIndex];
        console.log('当前分类数据:', category);
        console.log('分类名称:', currentLang === 'zh' ? category.category_zh : category.category_en);
        console.log('项目介绍存在:', !!(currentLang === 'zh' ? category.intro_zh : category.intro_en));
        console.log('项目介绍内容:', currentLang === 'zh' ? category.intro_zh : category.intro_en);
    }
    console.log('=== 调试结束 ===');
}

// 监听点击事件
document.addEventListener('click', (e) => {
    if (e.target.tagName === 'A' && e.target.classList.contains('top-nav-link')) {
        console.log('点击了导航链接:', e.target.textContent);
        setTimeout(debugRender, 100); // 等待render完成
    }
});

// 监听子分类点击
document.addEventListener('click', (e) => {
    if (e.target.tagName === 'A' && e.target.classList.contains('sub-nav-link')) {
        console.log('点击了子分类:', e.target.textContent);
        setTimeout(debugRender, 100); // 等待render完成
    }
});

// 初始调试
setTimeout(debugRender, 500); // 页面加载完成后调试

console.log('调试脚本已加载，点击导航时会输出调试信息');