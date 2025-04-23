// 错误日志页面JavaScript (Updated for new structure, no Bootstrap)

// 页面滚动功能
function scrollToTop() {
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function scrollToBottom() {
    window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
}

// Refresh function removed as the buttons are gone.
// If refresh functionality is needed elsewhere, it can be triggered directly by calling loadErrorLogs().

// 全局变量
let currentPage = 1;
let pageSize = 10;
// let totalPages = 1; // totalPages will be calculated dynamically based on API response if available, or based on fetched data length
let errorLogs = []; // Store fetched logs for details view
let currentSearch = { // Store current search parameters
    key: '',
    error: '',
    errorCode: '', // Added error code search
    startDate: '',
    endDate: ''
};

// DOM Elements Cache
let pageSizeSelector;
// let refreshBtn; // Removed, as the button is deleted
let tableBody;
let paginationElement;
let loadingIndicator;
let noDataMessage;
let errorMessage;
let logDetailModal;
let modalCloseBtns; // Collection of close buttons for the modal
let keySearchInput;
let errorSearchInput;
let errorCodeSearchInput; // Added error code input
let startDateInput;
let endDateInput;
let searchBtn;
let pageInput;
let goToPageBtn;
let selectAllCheckbox; // 新增：全选复选框
let copySelectedKeysBtn; // 新增：复制选中按钮
let selectedCountSpan; // 新增：选中计数显示

// 页面加载完成后执行
document.addEventListener('DOMContentLoaded', function() {
    // Cache DOM elements
    pageSizeSelector = document.getElementById('pageSize');
    // refreshBtn = document.getElementById('refreshBtn'); // Removed
    tableBody = document.getElementById('errorLogsTable');
    paginationElement = document.getElementById('pagination');
    loadingIndicator = document.getElementById('loadingIndicator');
    noDataMessage = document.getElementById('noDataMessage');
    errorMessage = document.getElementById('errorMessage');
    logDetailModal = document.getElementById('logDetailModal');
    // Get all elements that should close the modal
    modalCloseBtns = document.querySelectorAll('#closeLogDetailModalBtn, #closeModalFooterBtn');
    keySearchInput = document.getElementById('keySearch');
    errorSearchInput = document.getElementById('errorSearch');
    errorCodeSearchInput = document.getElementById('errorCodeSearch'); // Get error code input
    startDateInput = document.getElementById('startDate');
    endDateInput = document.getElementById('endDate');
    searchBtn = document.getElementById('searchBtn');
    pageInput = document.getElementById('pageInput');
    goToPageBtn = document.getElementById('goToPageBtn');
    selectAllCheckbox = document.getElementById('selectAllCheckbox'); // 新增
    copySelectedKeysBtn = document.getElementById('copySelectedKeysBtn'); // 新增
    selectedCountSpan = document.getElementById('selectedCount'); // 新增

    // Initialize page size selector
    if (pageSizeSelector) {
        pageSizeSelector.value = pageSize;
        pageSizeSelector.addEventListener('change', function() {
            pageSize = parseInt(this.value);
            currentPage = 1; // Reset to first page
            loadErrorLogs();
        });
    }

    // Refresh button event listener removed

    // Initialize search button
    if (searchBtn) {
        searchBtn.addEventListener('click', function() {
            // Update search parameters from input fields
            currentSearch.key = keySearchInput ? keySearchInput.value.trim() : '';
            currentSearch.error = errorSearchInput ? errorSearchInput.value.trim() : '';
            currentSearch.errorCode = errorCodeSearchInput ? errorCodeSearchInput.value.trim() : ''; // Get error code value
            currentSearch.startDate = startDateInput ? startDateInput.value : '';
            currentSearch.endDate = endDateInput ? endDateInput.value : '';
            currentPage = 1; // Reset to first page on new search
            loadErrorLogs();
        });
    }

    // Initialize modal close buttons
    if (logDetailModal && modalCloseBtns) {
        modalCloseBtns.forEach(btn => {
            btn.addEventListener('click', closeLogDetailModal);
        });
        // Optional: Close modal if clicking outside the content
        logDetailModal.addEventListener('click', function(event) {
            if (event.target === logDetailModal) {
                closeLogDetailModal();
            }
        });
    }

    // Initial load of error logs
    loadErrorLogs();

    // Add event listeners for copy buttons inside the modal and table
    setupCopyButtons(); // This will now also handle table copy buttons if called after render

    // Add event listeners for bulk selection
    setupBulkSelectionListeners(); // 新增：设置批量选择监听器

    // 新增：为页码跳转按钮添加事件监听器
    if (goToPageBtn && pageInput) {
        goToPageBtn.addEventListener('click', function() {
            const targetPage = parseInt(pageInput.value);
            // 需要获取总页数来验证输入
            // 暂时无法直接获取 totalPages，需要在 updatePagination 中存储或重新计算
            // 简单的验证：必须是正整数
            if (!isNaN(targetPage) && targetPage >= 1) {
                // 理想情况下，应检查 targetPage <= totalPages
                // 但 totalPages 可能未知，所以暂时只跳转
                currentPage = targetPage;
                loadErrorLogs();
                pageInput.value = ''; // 清空输入框
            } else {
                showNotification('请输入有效的页码', 'error', 2000);
                pageInput.value = ''; // 清空无效输入
            }
        });
        // 允许按 Enter 键跳转
        pageInput.addEventListener('keypress', function(event) {
            if (event.key === 'Enter') {
                goToPageBtn.click(); // 触发按钮点击
            }
        });
    }
});

// Fallback copy function using document.execCommand
function fallbackCopyTextToClipboard(text) {
    const textArea = document.createElement("textarea");
    textArea.value = text;

    // Avoid scrolling to bottom
    textArea.style.top = "0";
    textArea.style.left = "0";
    textArea.style.position = "fixed";

    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();

    let successful = false;
    try {
        successful = document.execCommand('copy');
    } catch (err) {
        console.error('Fallback copy failed:', err);
        successful = false;
    }

    document.body.removeChild(textArea);
    return successful;
}

// Helper function to handle feedback after copy attempt (both modern and fallback)
function handleCopyResult(buttonElement, success) {
     const originalIcon = buttonElement.querySelector('i').className; // Store original icon class
     const iconElement = buttonElement.querySelector('i');
     if (success) {
        iconElement.className = 'fas fa-check text-success-500'; // Use checkmark icon class
        showNotification('已复制到剪贴板', 'success', 2000);
     } else {
        iconElement.className = 'fas fa-times text-danger-500'; // Use error icon class
        showNotification('复制失败', 'error', 3000);
     }
     setTimeout(() => { iconElement.className = originalIcon; }, success ? 2000 : 3000); // Restore original icon class
}

// Function to set up copy button listeners (using modern API with fallback) - Updated to handle table copy buttons
function setupCopyButtons(containerSelector = 'body') {
    // Find buttons within the specified container (defaults to body)
    const container = document.querySelector(containerSelector);
    if (!container) return;

    const copyButtons = container.querySelectorAll('.copy-btn');
    copyButtons.forEach(button => {
        // Remove existing listener to prevent duplicates if called multiple times
        button.removeEventListener('click', handleCopyButtonClick);
        // Add the listener
        button.addEventListener('click', handleCopyButtonClick);
    });
}

// Extracted click handler logic for reusability and removing listeners
function handleCopyButtonClick() {
    const button = this; // 'this' refers to the button clicked
    const targetId = button.getAttribute('data-target');
    const textToCopyDirect = button.getAttribute('data-copy-text'); // For direct text copy (e.g., table key)
    let textToCopy = '';

    if (textToCopyDirect) {
        textToCopy = textToCopyDirect;
    } else if (targetId) {
        const targetElement = document.getElementById(targetId);
        if (targetElement) {
            textToCopy = targetElement.textContent;
        } else {
            console.error('Target element not found:', targetId);
            showNotification('复制出错：找不到目标元素', 'error');
            return; // Exit if target element not found
        }
    } else {
        console.error('No data-target or data-copy-text attribute found on button:', button);
        showNotification('复制出错：未指定复制内容', 'error');
        return; // Exit if no source specified
    }


    if (textToCopy) {
        let copySuccess = false;
        // Try modern clipboard API first (requires HTTPS or localhost)
        if (navigator.clipboard && window.isSecureContext) {
            navigator.clipboard.writeText(textToCopy).then(() => {
                handleCopyResult(button, true); // Use helper for feedback
            }).catch(err => {
                console.error('Clipboard API failed, attempting fallback:', err);
                // Attempt fallback if modern API fails
                copySuccess = fallbackCopyTextToClipboard(textToCopy);
                handleCopyResult(button, copySuccess); // Use helper for feedback
            });
        } else {
            // Use fallback if modern API is not available or context is insecure
            console.warn("Clipboard API not available or context insecure. Using fallback copy method.");
            copySuccess = fallbackCopyTextToClipboard(textToCopy);
            handleCopyResult(button, copySuccess); // Use helper for feedback
        }
    } else {
        console.warn('No text found to copy for target:', targetId || 'direct text');
        showNotification('没有内容可复制', 'warning');
    }
} // End of handleCopyButtonClick function

// Function to set up copy button listeners (using modern API with fallback) - Updated to handle table copy buttons
function setupCopyButtons(containerSelector = 'body') {
    // Find buttons within the specified container (defaults to body)
    const container = document.querySelector(containerSelector);
    if (!container) return;

    const copyButtons = container.querySelectorAll('.copy-btn');
    copyButtons.forEach(button => {
        // Remove existing listener to prevent duplicates if called multiple times
        button.removeEventListener('click', handleCopyButtonClick);
        // Add the listener
        button.addEventListener('click', handleCopyButtonClick);
    });
}

// 新增：设置批量选择相关的事件监听器
function setupBulkSelectionListeners() {
   if (selectAllCheckbox) {
       selectAllCheckbox.addEventListener('change', handleSelectAllChange);
   }

   if (tableBody) {
       // 使用事件委托处理行复选框的点击
       tableBody.addEventListener('change', handleRowCheckboxChange);
   }

   if (copySelectedKeysBtn) {
       copySelectedKeysBtn.addEventListener('click', handleCopySelectedKeys);
   }
}

// 新增：处理“全选”复选框变化的函数
function handleSelectAllChange() {
   const isChecked = selectAllCheckbox.checked;
   const rowCheckboxes = tableBody.querySelectorAll('.row-checkbox');
   rowCheckboxes.forEach(checkbox => {
       checkbox.checked = isChecked;
   });
   updateSelectedState();
}

// 新增：处理行复选框变化的函数 (事件委托)
function handleRowCheckboxChange(event) {
   if (event.target.classList.contains('row-checkbox')) {
       updateSelectedState();
   }
}

// 新增：更新选中状态（计数、按钮状态、全选框状态）
function updateSelectedState() {
   const rowCheckboxes = tableBody.querySelectorAll('.row-checkbox');
   const selectedCheckboxes = tableBody.querySelectorAll('.row-checkbox:checked');
   const selectedCount = selectedCheckboxes.length;

   // 移除了数字显示，不再更新selectedCountSpan
   // 仍然更新复制按钮的禁用状态
   if (copySelectedKeysBtn) {
       copySelectedKeysBtn.disabled = selectedCount === 0;
       
       // 可选：根据选中项数量更新按钮标题属性
       copySelectedKeysBtn.setAttribute('title', `复制${selectedCount}项选中密钥`);
   }

   // 更新“全选”复选框的状态
   if (selectAllCheckbox) {
       if (rowCheckboxes.length > 0 && selectedCount === rowCheckboxes.length) {
           selectAllCheckbox.checked = true;
           selectAllCheckbox.indeterminate = false;
       } else if (selectedCount > 0) {
           selectAllCheckbox.checked = false;
           selectAllCheckbox.indeterminate = true; // 部分选中状态
       } else {
           selectAllCheckbox.checked = false;
           selectAllCheckbox.indeterminate = false;
       }
   }
}

// 新增：处理“复制选中密钥”按钮点击的函数
function handleCopySelectedKeys() {
   const selectedCheckboxes = tableBody.querySelectorAll('.row-checkbox:checked');
   const keysToCopy = [];
   selectedCheckboxes.forEach(checkbox => {
       const key = checkbox.getAttribute('data-key');
       if (key) {
           keysToCopy.push(key);
       }
   });

   if (keysToCopy.length > 0) {
       const textToCopy = keysToCopy.join('\n'); // 每行一个密钥
       copyTextToClipboard(textToCopy, copySelectedKeysBtn); // 使用通用复制函数
   } else {
       showNotification('没有选中的密钥可复制', 'warning');
   }
}

// 新增：通用的文本复制函数（结合现有逻辑）
function copyTextToClipboard(text, buttonElement = null) {
   let copySuccess = false;
   if (navigator.clipboard && window.isSecureContext) {
       navigator.clipboard.writeText(text).then(() => {
           if (buttonElement) handleCopyResult(buttonElement, true);
           else showNotification('已复制到剪贴板', 'success');
       }).catch(err => {
           console.error('Clipboard API failed, attempting fallback:', err);
           copySuccess = fallbackCopyTextToClipboard(text);
           if (buttonElement) handleCopyResult(buttonElement, copySuccess);
           else showNotification(copySuccess ? '已复制到剪贴板' : '复制失败', copySuccess ? 'success' : 'error');
       });
   } else {
       console.warn("Clipboard API not available or context insecure. Using fallback copy method.");
       copySuccess = fallbackCopyTextToClipboard(text);
       if (buttonElement) handleCopyResult(buttonElement, copySuccess);
       else showNotification(copySuccess ? '已复制到剪贴板' : '复制失败', copySuccess ? 'success' : 'error');
   }
}


// 加载错误日志数据
async function loadErrorLogs() {
    // 重置选择状态
    if (selectAllCheckbox) selectAllCheckbox.checked = false;
    if (selectAllCheckbox) selectAllCheckbox.indeterminate = false;
    updateSelectedState(); // 更新按钮状态和计数

    showLoading(true);
    showError(false);
    showNoData(false);

    const offset = (currentPage - 1) * pageSize;

    try {
        // Construct the API URL with search parameters
        let apiUrl = `/api/logs/errors?limit=${pageSize}&offset=${offset}`;
        if (currentSearch.key) {
            apiUrl += `&key_search=${encodeURIComponent(currentSearch.key)}`;
        }
        if (currentSearch.error) {
            apiUrl += `&error_search=${encodeURIComponent(currentSearch.error)}`;
        }
        if (currentSearch.errorCode) { // Add error code to API request
            apiUrl += `&error_code_search=${encodeURIComponent(currentSearch.errorCode)}`;
        }
        if (currentSearch.startDate) {
            apiUrl += `&start_date=${encodeURIComponent(currentSearch.startDate)}`;
        }
        if (currentSearch.endDate) {
            apiUrl += `&end_date=${encodeURIComponent(currentSearch.endDate)}`;
        }

        const response = await fetch(apiUrl);
        if (!response.ok) {
            // Try to get error message from response body
            let errorData;
            try {
                errorData = await response.json();
            } catch (e) {
                // Ignore if response is not JSON
            }
            throw new Error(errorData?.detail || `网络响应异常: ${response.statusText}`);
        }
        const data = await response.json();
        // API 现在返回 { logs: [], total: count }
        if (data && Array.isArray(data.logs)) {
            errorLogs = data.logs; // Store the list data (contains error_code)
            renderErrorLogs(errorLogs);
            updatePagination(errorLogs.length, data.total || -1);
        } else {
            throw new Error('无法识别的API响应格式');
        }


        showLoading(false);

        if (errorLogs.length === 0) {
            showNoData(true);
        }
    } catch (error) {
        console.error('获取错误日志失败:', error);
        showLoading(false);
        showError(true, error.message); // Show specific error message
    }
}


// 渲染错误日志表格
function renderErrorLogs(logs) {
    if (!tableBody) return;
    tableBody.innerHTML = ''; // Clear previous entries

    // 重置全选复选框状态（在清空表格后）
    if (selectAllCheckbox) {
        selectAllCheckbox.checked = false;
        selectAllCheckbox.indeterminate = false;
    }

    if (!logs || logs.length === 0) {
        // Handled by showNoData
        return;
    }

    const startIndex = (currentPage - 1) * pageSize; // Calculate starting index for the current page

    logs.forEach((log, index) => { // Add index parameter to forEach
        const row = document.createElement('tr');
        const sequentialId = startIndex + index + 1; // Calculate sequential ID for the current page
        // Format date
        let formattedTime = 'N/A';
        try {
            const requestTime = new Date(log.request_time);
            if (!isNaN(requestTime)) {
                 formattedTime = requestTime.toLocaleString('zh-CN', {
                    year: 'numeric', month: '2-digit', day: '2-digit',
                    hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false
                });
            }
        } catch (e) { console.error("Error formatting date:", e); }


        // Display error code instead of truncated log
        const errorCodeContent = log.error_code || '无';

        // Mask the Gemini key for display in the table
        const maskKey = (key) => {
            if (!key || key.length < 8) return key || '无'; // Don't mask short keys or null
            return `${key.substring(0, 4)}...${key.substring(key.length - 4)}`;
        };
        const maskedKey = maskKey(log.gemini_key);
        const fullKey = log.gemini_key || ''; // Store the full key

        row.innerHTML = `
            <td class="text-center px-3 py-3"> <!-- Checkbox column -->
                <input type="checkbox" class="row-checkbox form-checkbox h-4 w-4 text-primary-600 border-gray-300 rounded focus:ring-primary-500" data-key="${fullKey}">
            </td>
            <td>${sequentialId}</td> <!-- Use sequential ID -->
            <td class="relative group" title="${fullKey}"> <!-- Added relative/group for button positioning -->
                ${maskedKey}
                <!-- Added copy button for the key in the table row -->
                <button class="copy-btn absolute top-1/2 right-2 transform -translate-y-1/2 bg-gray-200 hover:bg-gray-300 text-gray-600 p-1 rounded opacity-0 group-hover:opacity-100 transition-opacity text-xs" data-copy-text="${log.gemini_key || ''}" title="复制完整密钥">
                    <i class="far fa-copy"></i>
                </button>
            </td>
            <td>${log.error_type || '未知'}</td>
            <td class="error-code-content" title="${log.error_code || ''}">${errorCodeContent}</td>
            <td>${log.model_name || '未知'}</td>
            <td>${formattedTime}</td>
            <td>
                <button class="btn-view-details" data-log-id="${log.id}">
                    查看详情
                </button>
            </td>
        `;

        tableBody.appendChild(row);
    });

    // Add event listeners to new 'View Details' buttons
    document.querySelectorAll('.btn-view-details').forEach(button => {
        button.addEventListener('click', function() {
            const logId = parseInt(this.getAttribute('data-log-id'));
            showLogDetails(logId);
        });
    });

    // Re-initialize copy buttons specifically for the newly rendered table rows
    setupCopyButtons('#errorLogsTable');
    // Update selected state after rendering
    updateSelectedState();
}

// 显示错误日志详情 (从 API 获取)
async function showLogDetails(logId) {
    if (!logDetailModal) return;

    // Show loading state in modal (optional)
    // Clear previous content and show a spinner or message
    document.getElementById('modalGeminiKey').textContent = '加载中...';
    document.getElementById('modalErrorType').textContent = '加载中...';
    document.getElementById('modalErrorLog').textContent = '加载中...';
    document.getElementById('modalRequestMsg').textContent = '加载中...';
    document.getElementById('modalModelName').textContent = '加载中...';
    document.getElementById('modalRequestTime').textContent = '加载中...';

    logDetailModal.classList.add('show');
    document.body.style.overflow = 'hidden'; // Prevent body scrolling

    try {
        const response = await fetch(`/api/logs/errors/${logId}/details`);
        if (!response.ok) {
            let errorData;
            try {
                errorData = await response.json();
            } catch (e) { /* ignore */ }
            throw new Error(errorData?.detail || `获取日志详情失败: ${response.statusText}`);
        }
        const logDetails = await response.json();

        // Format date
        let formattedTime = 'N/A';
        try {
            const requestTime = new Date(logDetails.request_time);
            if (!isNaN(requestTime)) {
                formattedTime = requestTime.toLocaleString('zh-CN', {
                    year: 'numeric', month: '2-digit', day: '2-digit',
                    hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false
                });
            }
        } catch (e) { console.error("Error formatting date:", e); }

        // Format request message (handle potential JSON)
        let formattedRequestMsg = '无';
        if (logDetails.request_msg) {
            try {
                if (typeof logDetails.request_msg === 'object' && logDetails.request_msg !== null) {
                    formattedRequestMsg = JSON.stringify(logDetails.request_msg, null, 2);
                } else if (typeof logDetails.request_msg === 'string') {
                    // Try parsing if it looks like JSON, otherwise display as string
                    const trimmedMsg = logDetails.request_msg.trim();
                    if (trimmedMsg.startsWith('{') || trimmedMsg.startsWith('[')) {
                        formattedRequestMsg = JSON.stringify(JSON.parse(logDetails.request_msg), null, 2);
                    } else {
                        formattedRequestMsg = logDetails.request_msg;
                    }
                } else {
                     formattedRequestMsg = String(logDetails.request_msg);
                }
            } catch (e) {
                formattedRequestMsg = String(logDetails.request_msg); // Fallback
                console.warn("Could not parse request_msg as JSON:", e);
            }
        }

        // Populate modal content with fetched details
        document.getElementById('modalGeminiKey').textContent = logDetails.gemini_key || '无';
        document.getElementById('modalErrorType').textContent = logDetails.error_type || '未知';
        document.getElementById('modalErrorLog').textContent = logDetails.error_log || '无'; // Full error log
        document.getElementById('modalRequestMsg').textContent = formattedRequestMsg; // Full request message
        document.getElementById('modalModelName').textContent = logDetails.model_name || '未知';
        document.getElementById('modalRequestTime').textContent = formattedTime;

        // Re-initialize copy buttons specifically for the modal after content is loaded
        setupCopyButtons('#logDetailModal');

    } catch (error) {
        console.error('获取日志详情失败:', error);
        // Show error in modal
        document.getElementById('modalGeminiKey').textContent = '错误';
        document.getElementById('modalErrorType').textContent = '错误';
        document.getElementById('modalErrorLog').textContent = `加载失败: ${error.message}`;
        document.getElementById('modalRequestMsg').textContent = '错误';
        document.getElementById('modalModelName').textContent = '错误';
        document.getElementById('modalRequestTime').textContent = '错误';
        // Optionally show a notification
        showNotification(`加载日志详情失败: ${error.message}`, 'error', 5000);
    }
}

// Close Log Detail Modal
function closeLogDetailModal() {
     if (logDetailModal) {
        logDetailModal.classList.remove('show');
        // Optional: Restore body scrolling
        document.body.style.overflow = '';
     }
}


// 更新分页控件
function updatePagination(currentItemCount, totalItems) {
    if (!paginationElement) return;
    paginationElement.innerHTML = ''; // Clear existing pagination

    // Calculate total pages only if totalItems is known and valid
    let totalPages = 1;
    if (totalItems >= 0) {
        totalPages = Math.max(1, Math.ceil(totalItems / pageSize));
    } else if (currentItemCount < pageSize && currentPage === 1) {
        // If less items than page size fetched on page 1, assume it's the only page
        totalPages = 1;
    } else {
        // If total is unknown and more items might exist, we can't build full pagination
        // We can show Prev/Next based on current page and if items were returned
        console.warn("Total item count unknown, pagination will be limited.");
        // Basic Prev/Next for unknown total
        addPaginationLink(paginationElement, '&laquo;', currentPage > 1, () => { currentPage--; loadErrorLogs(); });
        addPaginationLink(paginationElement, currentPage.toString(), true, null, true); // Current page number (non-clickable)
        addPaginationLink(paginationElement, '&raquo;', currentItemCount === pageSize, () => { currentPage++; loadErrorLogs(); }); // Next enabled if full page was returned
        return; // Exit here for limited pagination
    }


    const maxPagesToShow = 5; // Max number of page links to show
    let startPage = Math.max(1, currentPage - Math.floor(maxPagesToShow / 2));
    let endPage = Math.min(totalPages, startPage + maxPagesToShow - 1);

    // Adjust startPage if endPage reaches the limit first
     if (endPage === totalPages) {
        startPage = Math.max(1, endPage - maxPagesToShow + 1);
    }


    // Previous Button
    addPaginationLink(paginationElement, '&laquo;', currentPage > 1, () => { currentPage--; loadErrorLogs(); });

    // First Page Button
    if (startPage > 1) {
        addPaginationLink(paginationElement, '1', true, () => { currentPage = 1; loadErrorLogs(); });
        if (startPage > 2) {
            addPaginationLink(paginationElement, '...', false); // Ellipsis
        }
    }

    // Page Number Buttons
    for (let i = startPage; i <= endPage; i++) {
        addPaginationLink(paginationElement, i.toString(), true, () => { currentPage = i; loadErrorLogs(); }, i === currentPage);
    }

     // Last Page Button
    if (endPage < totalPages) {
        if (endPage < totalPages - 1) {
             addPaginationLink(paginationElement, '...', false); // Ellipsis
        }
        addPaginationLink(paginationElement, totalPages.toString(), true, () => { currentPage = totalPages; loadErrorLogs(); });
    }


    // Next Button
    addPaginationLink(paginationElement, '&raquo;', currentPage < totalPages, () => { currentPage++; loadErrorLogs(); });
}

// Helper function to add pagination links
function addPaginationLink(parentElement, text, enabled, clickHandler, isActive = false) {
    const pageItem = document.createElement('li');
    // 移除 'page-item' 和 'active' 类，使用 Tailwind 类进行样式化
    // pageItem.className = `page-item ${!enabled ? 'disabled' : ''} ${isActive ? 'active' : ''}`;

    const pageLink = document.createElement('a');
    // 使用 Tailwind 类进行样式化
    pageLink.className = `px-3 py-1 rounded-md text-sm transition duration-150 ease-in-out ${
        isActive
            ? 'bg-primary-600 text-white font-semibold shadow-md cursor-default' // 突出当前页样式
            : enabled
                ? 'bg-white text-gray-700 hover:bg-primary-50 hover:text-primary-600 border border-gray-300' // 可点击页码样式
                : 'bg-gray-100 text-gray-400 cursor-not-allowed border border-gray-200' // 禁用状态样式 (如 '...')
    }`;
    pageLink.href = '#'; // Prevent page jump
    pageLink.innerHTML = text;

    if (enabled && clickHandler) {
        pageLink.addEventListener('click', function(e) {
            e.preventDefault();
            clickHandler();
        });
    } else if (!enabled) {
         pageLink.addEventListener('click', e => e.preventDefault()); // Prevent click on disabled or active
    } else if (isActive) {
        pageLink.addEventListener('click', e => e.preventDefault()); // Prevent click on active page
    }

    // 不再需要 li 元素，直接将 a 元素添加到父元素
    // pageItem.appendChild(pageLink);
    parentElement.appendChild(pageLink);
}


// 显示/隐藏状态指示器 (using 'active' class)
function showLoading(show) {
    if (loadingIndicator) loadingIndicator.style.display = show ? 'block' : 'none';
}

function showNoData(show) {
     if (noDataMessage) noDataMessage.style.display = show ? 'block' : 'none';
}

function showError(show, message = '加载错误日志失败，请稍后重试。') {
    if (errorMessage) {
        errorMessage.style.display = show ? 'block' : 'none';
        if (show) {
            // Update the error message content
            const p = errorMessage.querySelector('p');
            if (p) p.textContent = message;
        }
    }
}

// Function to show temporary status notifications (like copy success)
function showNotification(message, type = 'success', duration = 3000) {
    const notificationElement = document.getElementById('notification'); // Use the correct ID from base.html
    if (!notificationElement) {
        console.error("Notification element with ID 'notification' not found.");
        return;
    }

    // Set message and type class
    notificationElement.textContent = message;
    // Remove previous type classes before adding the new one
    notificationElement.classList.remove('success', 'error', 'warning', 'info');
    notificationElement.classList.add(type); // Add the type class for styling
    notificationElement.className = `notification ${type} show`; // Add 'show' class

    // Hide after duration
    setTimeout(() => {
        notificationElement.classList.remove('show');
    }, duration);
}

// Example Usage (if copy functionality is added later):
// showNotification('密钥已复制!', 'success');
// showNotification('复制失败!', 'error');
