// Wait for MathJax to be ready before rendering
document.addEventListener('DOMContentLoaded', () => {
    const waitForMathJax = () => {
        if (window.MathJax && window.MathJax.startup && window.MathJax.startup.document) {
            // MathJax is ready
            initPage();
        } else {
            // Check again in 100ms
            setTimeout(waitForMathJax, 100);
        }
    };

    waitForMathJax();
});

function initPage() {
    let currentLang = 'zh'; // 'zh' or 'en'
    let currentTopCategory = 'quickIndex'; // 'quickIndex', 'datasetScripts', 'toolScripts', 'trainingScripts', 'faq'
    let currentSubCategoryIndex = 0; // 用于数据集脚本的子分类索引
    
    // FAQ Data
    const faqData = [
        {
            q_zh: "1. 与 Grokking 现象的关系？",
            q_en: "1. Relation to the Grokking phenomenon?",
            a_zh: "虽然目前尚未能完全阐释 Grokking 现象的内在机理，但本研究倾向于认为，应当存在一个统一的神经网络学习理论框架，能够同时涵盖本研究提出的“神经雕刻范式”与 Grokking 现象。",
            a_en: "Although the mechanism behind the Grokking phenomenon has yet to be fully elucidated, this research suggests the existence of a unified theoretical framework for neural network learning that can simultaneously explain both the \"Neural Sculpting Paradigm\" proposed herein and Grokking."
        },
        {
            q_zh: "2. 与 Othello-GPT 的关系？",
            q_en: "2. Relation to Othello-GPT?",
            a_zh: "本研究发现，该方法的适用性不仅超越了简单任务的范畴，且不局限于自回归 Transformer 架构。实验证据表明，精确规则学习能力是神经网络的一种内生本质属性。",
            a_en: "This study reveals that the applicability of the proposed method extends beyond simple tasks and is not confined to autoregressive Transformer architectures. Experimental evidence indicates that the capacity for precise rule learning is an intrinsic property of neural networks."
        },
        {
            q_zh: "3. MNIST 实验说明了什么？",
            q_en: "3. What does the MNIST experiment demonstrate?",
            a_zh: "实验表明，神经网络表现为模式识别还是逻辑推理，主要取决于训练数据的特性，而这两种能力均为神经网络的固有潜能。这一发现有力地暗示了许多传统模式识别任务背后可能隐藏着潜在的逻辑幽灵，同时也支持了大型语言模型（LLM）确实习得了某种形式的“逻辑”这一观点。因此，将 LLM 简单视为“随机鹦鹉”并不恰当，但将其视为具有自主意识并理解自身言论的个体同样有失偏颇。",
            a_en: "Experiments demonstrate that whether a neural network exhibits pattern recognition or logical reasoning depends primarily on the characteristics of the training data, and both capabilities are inherent potentials of neural networks. This finding strongly suggests that a latent \"ghost of logic\" may underlie many traditional pattern recognition tasks and supports the view that Large Language Models (LLMs) have indeed acquired some form of \"logic.\" Consequently, viewing LLMs merely as \"stochastic parrots\" is inadequate, yet regarding them as conscious entities with genuine understanding would also be inaccurate."
        },
        {
            q_zh: "4. 与“大模型写代码 + Python解释器”有什么不同？",
            q_en: "4. Difference from 'LLM writing code + Python interpreter'?",
            a_zh: "LLM 生成代码展示了从自然语言描述到形式化代码的映射能力，但这依赖于外部解释器来执行代码。相比之下，本范式旨在通过端到端的训练，使神经网络直接内化执行逻辑。此外，仅通过 LLM 生成代码的方法，难以从输入输出数据集中训练出一个能够对未见数据进行精确预测的独立神经网络模型。",
            a_en: "LLM code generation demonstrates the capability to map natural language descriptions to formal code, success in which still relies on external interpreters for execution. In contrast, this paradigm aims to enable neural networks to internalize execution logic directly through end-to-end training. Furthermore, implied by reliance on code generation, it is difficult to train an independent neural network capable of accurately predicting unseen outputs solely from input-output datasets."
        },
        {
            q_zh: "5. 关于架构的选择",
            q_en: "5. About architecture choice",
            a_zh: "选择多层感知机（MLP）主要是为了最直观地证明该范式的通用性。事实上，多种模型架构均能胜任此类任务。本研究目前的重点在于验证可行性，尚未对不同架构的计算效率进行系统性的对比分析。",
            a_en: "The choice of Multi-Layer Perceptrons (MLP) was primarily to demonstrate the universality of the paradigm in the most straightforward manner. In fact, various architectures are capable of performing such tasks. The current focus of this research is on feasibility verification, and a systematic comparative analysis of computational efficiency across different architectures has not yet been conducted."
        },
        {
            q_zh: "6. 之前的应用是否可以用这个范式解释？",
            q_en: "6. Can previous applications be explained by this paradigm?",
            a_zh: "NeRF, RenderFormer 以及 DeepMind 的 \"Amortized Planning\" (Chess Transformer) 等工作在理念上与本范式较为接近。\n\n值得注意的是，本研究的大部分探索集中于输入输出均为离散表示的任务，但这并非必要条件（如上述提到的相关工作）。这表明，本范式的核心在于分布式表示、分布式计算以及精确变换规则的学习，而非受限于数据的离散或连续形式。",
            a_en: "Works such as NeRF, RenderFormer, and DeepMind's \"Amortized Planning\" (Chess Transformer) align closely with the philosophy of this paradigm.\n\nIt is worth noting that while a significant portion of this research focuses on tasks with discrete input/output representations, this is not a prerequisite (as evidenced by the aforementioned related works). This underscores that the core of this paradigm lies in distributed representation, distributed computation, and the learning of precise transformation rules, rather than being constrained by the discrete or continuous nature of the data."
        },
        {
            q_zh: "7. 这难道不是在“记忆”训练集吗？",
            q_en: "7. Is this just 'memorizing' the training set?",
            a_zh: "并非简单的记忆。尽管本研究使用了较大规模的模型和相对较小的数据集，但在整个输入空间中未曾见过的验证集上，模型依然保持了极高的测试精度，证明了其确实拟合数据集背后的变换规则，具备真正的泛化能力。",
            a_en: "It is not mere memorization. Despite utilizing relatively large models with small datasets, the models achieve extremely high test accuracy on validation sets drawn from the full input space that were never seen during training, proving that they have indeed fitted the transformation rules underlying the dataset and possess genuine generalization capability."
        }
    ];

    const langToggleBtn = document.getElementById('lang-toggle');
    const categoryNav = document.getElementById('category-nav');
    const scriptsList = document.getElementById('scripts-list');

    // Initialize
    render();

    // Event Listeners
    langToggleBtn.addEventListener('click', () => {
        currentLang = currentLang === 'zh' ? 'en' : 'zh';
        updateLangButton();
        render();
    });

    function updateLangButton() {
        langToggleBtn.textContent = currentLang === 'zh' ? 'Switch to English' : '切换到中文';
        document.documentElement.lang = currentLang === 'zh' ? 'zh-CN' : 'en';
    }

    function render() {
        // Clear existing content
        categoryNav.innerHTML = '';
        scriptsList.innerHTML = '';

        // ========== 渲染顶层分类导航 (三大类 + FAQ) ==========
        const topNavContainer = document.createElement('div');
        topNavContainer.className = 'top-nav-container';
        
        const topNavUl = document.createElement('ul');
        topNavUl.className = 'top-nav';
        
        const topCategories = [
            { key: 'quickIndex', name_zh: '快速索引', name_en: 'Quick Index' },
            { key: 'datasetScripts', name_zh: docsData.datasetScripts.name_zh, name_en: docsData.datasetScripts.name_en },
            { key: 'researchProjects', name_zh: docsData.researchProjects.name_zh, name_en: docsData.researchProjects.name_en },
            { key: 'independentProjects', name_zh: docsData.independentProjects.name_zh, name_en: docsData.independentProjects.name_en },
            { key: 'toolScripts', name_zh: docsData.toolScripts.name_zh, name_en: docsData.toolScripts.name_en },
            { key: 'trainingScripts', name_zh: docsData.trainingScripts.name_zh, name_en: docsData.trainingScripts.name_en },
            { key: 'faq', name_zh: '常见问题', name_en: 'FAQ' } // Added FAQ
        ];
        
        topCategories.forEach(cat => {
            const li = document.createElement('li');
            const a = document.createElement('a');
            a.href = '#';
            a.textContent = currentLang === 'zh' ? cat.name_zh : cat.name_en;
            a.className = 'top-nav-link';
            
            if (cat.key === currentTopCategory) {
                a.classList.add('active');
            }
            
            a.addEventListener('click', (e) => {
                e.preventDefault();
                currentTopCategory = cat.key;
                currentSubCategoryIndex = 0; // 重置子分类索引
                render();
            });
            
            li.appendChild(a);
            topNavUl.appendChild(li);
        });
        
        topNavContainer.appendChild(topNavUl);
        categoryNav.appendChild(topNavContainer);

        // ========== 渲染子分类导航 (数据集脚本、研究项目和独立项目有子分类) ==========
        if (currentTopCategory === 'datasetScripts' || currentTopCategory === 'researchProjects' || currentTopCategory === 'independentProjects') {
            const subNavContainer = document.createElement('div');
            subNavContainer.className = 'sub-nav-container';
            
            const subNavUl = document.createElement('ul');
            subNavUl.className = 'sub-nav';
            
            const categories = docsData[currentTopCategory].categories;
            
            categories.forEach((category, index) => {
                const li = document.createElement('li');
                const a = document.createElement('a');
                a.href = '#';
                a.textContent = currentLang === 'zh' ? category.category_zh : category.category_en;
                a.className = 'sub-nav-link';
                
                if (index === currentSubCategoryIndex) {
                    a.classList.add('active');
                }
                
                a.addEventListener('click', (e) => {
                    e.preventDefault();
                    currentSubCategoryIndex = index;
                    renderMainContent();
                    updateSubNav();
                });
                
                li.appendChild(a);
                subNavUl.appendChild(li);
            });
            
            subNavContainer.appendChild(subNavUl);
            categoryNav.appendChild(subNavContainer);
        }

        // 渲染主要内容
        renderMainContent();

        function renderMainContent() {
            scriptsList.innerHTML = '';

            // ============ QUICK INDEX RENDERING ============
            if (currentTopCategory === 'quickIndex') {
                renderQuickIndex();
                return;
            }
            
            // ============ FAQ RENDERING ============
            if (currentTopCategory === 'faq') {
                const section = document.createElement('section');
                section.className = 'category-section';

                const title = document.createElement('h2');
                title.className = 'category-title';
                title.textContent = currentLang === 'zh' ? '常见问题' : 'Frequently Asked Questions';
                section.appendChild(title);

                faqData.forEach(item => {
                    const card = document.createElement('div');
                    card.className = 'script-card'; // Reuse script-card style

                    const header = document.createElement('div');
                    header.className = 'script-header';
                    header.style.justifyContent = 'flex-start'; // Align title to start
                    
                    const qTitle = document.createElement('h3');
                    qTitle.style.margin = '0';
                    qTitle.style.fontSize = '1.1em';
                    qTitle.textContent = currentLang === 'zh' ? item.q_zh : item.q_en;
                    
                    header.appendChild(qTitle);

                    const desc = document.createElement('div');
                    desc.className = 'script-desc';
                    // Allow newlines in answer
                    const answerText = currentLang === 'zh' ? item.a_zh : item.a_en;
                    desc.innerHTML = answerText.replace(/\n/g, '<br/>');

                    card.appendChild(header);
                    card.appendChild(desc);
                    section.appendChild(card);
                });

                scriptsList.appendChild(section);

                // Re-render MathJax after adding FAQ content
                if (window.MathJax) {
                    MathJax.typesetPromise([scriptsList]).catch((err) => console.error('MathJax typeset error:', err));
                }
                return;
            }

            // ============ SCRIPT RENDERING ============
            let scripts = [];
            let sectionTitle = '';

            if (currentTopCategory === 'datasetScripts' || currentTopCategory === 'researchProjects' || currentTopCategory === 'independentProjects') {
                const category = docsData[currentTopCategory].categories[currentSubCategoryIndex];
                if (category) {
                    scripts = category.scripts;
                    sectionTitle = currentLang === 'zh' ? category.category_zh : category.category_en;
                }
            } else if (currentTopCategory === 'toolScripts') {
                scripts = docsData.toolScripts.scripts;
                sectionTitle = currentLang === 'zh' ? docsData.toolScripts.name_zh : docsData.toolScripts.name_en;
            } else if (currentTopCategory === 'trainingScripts') {
                scripts = docsData.trainingScripts.scripts;
                sectionTitle = currentLang === 'zh' ? docsData.trainingScripts.name_zh : docsData.trainingScripts.name_en;
            }
            
            if (!scripts || scripts.length === 0) {
                const emptyMsg = document.createElement('p');
                emptyMsg.className = 'empty-message';
                emptyMsg.textContent = currentLang === 'zh' ? '暂无脚本' : 'No scripts available';
                scriptsList.appendChild(emptyMsg);
                return;
            }

            const section = document.createElement('section');
            section.className = 'category-section';

            const title = document.createElement('h2');
            title.className = 'category-title';
            title.textContent = sectionTitle;
            section.appendChild(title);

            // Render Introduction if available (For Research Projects and Independent Projects)
            if (currentTopCategory === 'researchProjects' || currentTopCategory === 'independentProjects') {
                const category = docsData[currentTopCategory].categories[currentSubCategoryIndex];
                const intro = currentLang === 'zh' ? category.intro_zh : category.intro_en;
                if (intro) {
                    const introDiv = document.createElement('div');
                    introDiv.className = 'category-intro';
                    // Use marked.js to render markdown content
                    introDiv.innerHTML = marked.parse(intro);
                    section.appendChild(introDiv);

                    // Re-render MathJax after adding content
                    if (window.MathJax) {
                        MathJax.typesetPromise([introDiv]).catch((err) => console.error('MathJax typeset error:', err));
                    }
                }
            }

            scripts.forEach(script => {
                const card = document.createElement('div');
                card.className = 'script-card';

                const header = document.createElement('div');
                header.className = 'script-header';
                
                const path = document.createElement('div');
                path.className = 'script-path';
                path.textContent = script.path;
                
                const expandBtn = document.createElement('button');
                expandBtn.className = 'expand-btn';
                expandBtn.textContent = currentLang === 'zh' ? '展开详情' : 'View Details';
                
                header.appendChild(path);
                header.appendChild(expandBtn);

                const desc = document.createElement('div');
                desc.className = 'script-desc';
                desc.textContent = currentLang === 'zh' ? script.description_zh : script.description_en;

                const details = document.createElement('div');
                details.className = 'script-details';
                // Use marked.js to render markdown content
                const rawDetails = currentLang === 'zh' ? script.details_zh : script.details_en;
                if (rawDetails) {
                    details.innerHTML = marked.parse(rawDetails);
                }

                // Re-render MathJax after adding content
                if (window.MathJax) {
                    MathJax.typesetPromise([details]).catch((err) => console.error('MathJax typeset error:', err));
                }

                // Expand/Collapse Logic
                expandBtn.addEventListener('click', () => {
                    const isActive = details.classList.contains('active');
                    if (isActive) {
                        details.classList.remove('active');
                        expandBtn.textContent = currentLang === 'zh' ? '展开详情' : 'View Details';
                    } else {
                        details.classList.add('active');
                        expandBtn.textContent = currentLang === 'zh' ? '收起详情' : 'Hide Details';
                    }
                });

                card.appendChild(header);
                card.appendChild(desc);
                card.appendChild(details);
                section.appendChild(card);
            });

            scriptsList.appendChild(section);
        }

        function renderQuickIndex() {
            const entries = collectIndexEntries();
            const section = document.createElement('section');
            section.className = 'category-section quick-index-section';

            const title = document.createElement('h2');
            title.className = 'category-title';
            title.textContent = currentLang === 'zh' ? '快速索引' : 'Quick Index';
            section.appendChild(title);

            const intro = document.createElement('p');
            intro.className = 'quick-index-intro';
            intro.textContent = currentLang === 'zh'
                ? `当前收录 ${entries.length} 个公开脚本。可以按名称、用途或分类搜索，点击脚本即可查看详细说明。`
                : `${entries.length} public scripts are indexed. Search by name, purpose, or category, then click a script to view its documentation.`;
            section.appendChild(intro);

            const controls = document.createElement('div');
            controls.className = 'quick-index-controls';

            const searchInput = document.createElement('input');
            searchInput.className = 'quick-index-search';
            searchInput.type = 'search';
            searchInput.placeholder = currentLang === 'zh' ? '搜索脚本、用途或分类...' : 'Search scripts, purposes, or categories...';
            searchInput.setAttribute('aria-label', currentLang === 'zh' ? '搜索脚本' : 'Search scripts');

            const typeFilter = document.createElement('select');
            typeFilter.className = 'quick-index-filter';
            typeFilter.setAttribute('aria-label', currentLang === 'zh' ? '按脚本类型筛选' : 'Filter by script type');
            [
                ['all', currentLang === 'zh' ? '全部类型' : 'All types'],
                ['datasetScripts', currentLang === 'zh' ? '数据集脚本' : 'Dataset scripts'],
                ['researchProjects', currentLang === 'zh' ? '研究项目' : 'Research projects'],
                ['independentProjects', currentLang === 'zh' ? '独立项目' : 'Independent projects'],
                ['toolScripts', currentLang === 'zh' ? '工具脚本' : 'Tool scripts'],
                ['trainingScripts', currentLang === 'zh' ? '训练脚本' : 'Training scripts']
            ].forEach(([value, label]) => {
                const option = document.createElement('option');
                option.value = value;
                option.textContent = label;
                typeFilter.appendChild(option);
            });

            const resultCount = document.createElement('span');
            resultCount.className = 'quick-index-count';

            controls.appendChild(searchInput);
            controls.appendChild(typeFilter);
            controls.appendChild(resultCount);
            section.appendChild(controls);

            const results = document.createElement('div');
            results.className = 'quick-index-results';
            section.appendChild(results);
            scriptsList.appendChild(section);

            const renderResults = () => {
                const query = searchInput.value.trim().toLowerCase();
                const type = typeFilter.value;
                const filtered = entries.filter(entry => {
                    const searchable = `${entry.path} ${entry.description_zh || ''} ${entry.description_en || ''} ${entry.group_zh} ${entry.group_en}`.toLowerCase();
                    return (type === 'all' || entry.topCategory === type) && (!query || searchable.includes(query));
                });

                resultCount.textContent = currentLang === 'zh'
                    ? `${filtered.length} 个结果`
                    : `${filtered.length} results`;
                results.innerHTML = '';

                if (filtered.length === 0) {
                    const empty = document.createElement('p');
                    empty.className = 'empty-message quick-index-empty';
                    empty.textContent = currentLang === 'zh' ? '没有匹配的脚本。' : 'No matching scripts.';
                    results.appendChild(empty);
                    return;
                }

                const groups = new Map();
                filtered.forEach(entry => {
                    const groupName = currentLang === 'zh' ? entry.group_zh : entry.group_en;
                    if (!groups.has(groupName)) groups.set(groupName, []);
                    groups.get(groupName).push(entry);
                });

                groups.forEach((groupEntries, groupName) => {
                    const group = document.createElement('section');
                    group.className = 'quick-index-group';

                    const heading = document.createElement('h3');
                    heading.className = 'quick-index-group-title';
                    heading.textContent = `${groupName} (${groupEntries.length})`;
                    group.appendChild(heading);

                    const list = document.createElement('ul');
                    list.className = 'quick-index-list';
                    groupEntries.forEach(entry => {
                        const item = document.createElement('li');
                        item.className = 'quick-index-item';

                        const button = document.createElement('button');
                        button.type = 'button';
                        button.className = 'quick-index-link';
                        button.textContent = entry.path;
                        button.addEventListener('click', () => openIndexedScript(entry));

                        const desc = document.createElement('span');
                        desc.className = 'quick-index-description';
                        desc.textContent = stripMarkdownForIndex(
                            currentLang === 'zh' ? entry.description_zh : entry.description_en
                        );

                        item.appendChild(button);
                        item.appendChild(desc);
                        list.appendChild(item);
                    });

                    group.appendChild(list);
                    results.appendChild(group);
                });
            };

            searchInput.addEventListener('input', renderResults);
            typeFilter.addEventListener('change', renderResults);
            renderResults();
        }

        function stripMarkdownForIndex(text = '') {
            return text
                .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
                .replace(/<[^>]+>/g, '')
                .replace(/^\s*[-+>]\s+/gm, '')
                .replace(/[*`~]/g, '')
                .replace(/\s+/g, ' ')
                .trim();
        }

        function collectIndexEntries() {
            const entries = [];
            const addCategories = topCategory => {
                docsData[topCategory].categories.forEach((category, subCategoryIndex) => {
                    category.scripts.forEach(script => entries.push({
                        ...script,
                        sourcePath: script.path,
                        path: normalizeIndexPath(script.path, topCategory, category.category_en),
                        topCategory,
                        subCategoryIndex,
                        group_zh: category.category_zh,
                        group_en: category.category_en
                    }));
                });
            };
            const addFlat = topCategory => {
                docsData[topCategory].scripts.forEach(script => entries.push({
                    ...script,
                    sourcePath: script.path,
                    path: normalizeIndexPath(script.path, topCategory),
                    topCategory,
                    subCategoryIndex: 0,
                    group_zh: docsData[topCategory].name_zh,
                    group_en: docsData[topCategory].name_en
                }));
            };

            addCategories('datasetScripts');
            addCategories('researchProjects');
            addCategories('independentProjects');
            addFlat('toolScripts');
            addFlat('trainingScripts');
            return entries;
        }

        function normalizeIndexPath(scriptPath, topCategory, categoryName = '') {
            const cleanPath = scriptPath.replace(/^\*+/, '');
            if (cleanPath.includes('/')) return cleanPath;

            if (topCategory === 'researchProjects') {
                const researchDirs = {
                    'NTK Batch Solver': 'research/ntk_batch_solver',
                    'Rule Preference Phase Transition': 'research/rule_preference',
                    'Meta-CA Learning Experiments': 'research/rule_ood_generalization'
                };
                return `${researchDirs[categoryName]}/${cleanPath}`;
            }

            if (topCategory === 'independentProjects') {
                const projectDirs = {
                    'Neural Processor': 'neural_processor',
                    'Neural Inverse Engineering': 'neural_inverse_engineering'
                };
                return projectDirs[categoryName]
                    ? `${projectDirs[categoryName]}/${cleanPath}`
                    : cleanPath;
            }

            return cleanPath;
        }

        function openIndexedScript(entry) {
            currentTopCategory = entry.topCategory;
            currentSubCategoryIndex = entry.subCategoryIndex;
            render();

            const cards = Array.from(scriptsList.querySelectorAll('.script-card'));
            const card = cards.find(item => item.querySelector('.script-path')?.textContent === entry.sourcePath);
            if (!card) return;

            const details = card.querySelector('.script-details');
            const expandBtn = card.querySelector('.expand-btn');
            if (details && expandBtn) {
                details.classList.add('active');
                expandBtn.textContent = currentLang === 'zh' ? '收起详情' : 'Hide Details';
            }
            card.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }

        function updateSubNav() {
            const subNavLinks = categoryNav.querySelectorAll('.sub-nav-link');
            subNavLinks.forEach((link, index) => {
                if (index === currentSubCategoryIndex) {
                    link.classList.add('active');
                } else {
                    link.classList.remove('active');
                }
            });
        }
    }
}
