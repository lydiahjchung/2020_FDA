function getPortfolioHTML(pfo_idx) {
    return `
        <li id="pfo_wrapper_${pfo_idx}" class="pfo" style="margin-top: 20px;">
            <div class="form-row">
                Portfolio Name : 
                <input type="text" class="pfo_name" name="pfo_name" placeholder="Portfolio Name" style="width:220px;"/>
            </div>
            <button type="button" class="add_asset_btn">Add Row</button>
            <button type="button" class="delete_asset_btn">Delete row</button>
            <ol class="asset_list">
                <li class="asset_wrapper">
                    <input type="text" class="asset asset_ticker" name="asset_ticker" placeholder="Ticker"/>
                    <input type="text" class="asset asset_name" name="asset_name" placeholder="Asset Name"/>
                    <input type="text" class="asset asset_weight" name="asset_weight" placeholder="Weight"/>
                    <input type="text" class="asset asset_leverage" name="asset_leverage" placeholder="Leverage"/>
                </li>
            </ol>
        </li>
    `;
}

function getAssetHTML() {
    return `
        <li class="asset_wrapper">
            <input type="text" class="asset asset_ticker" name="asset_ticker" placeholder="Ticker"/>
            <input type="text" class="asset asset_name" name="asset_name" placeholder="Asset Name"/>
            <input type="text" class="asset asset_weight" name="asset_weight" placeholder="Weight"/>
            <input type="text" class="asset asset_leverage" name="asset_leverage" placeholder="Leverage"/>
        </li>
    `;
}

function add_portfolio() {
    const pfo_div = document.getElementById("pfo_div");
    const new_pfo_idx = pfo_div.querySelectorAll('li').length;
    pfo_div.insertAdjacentHTML('beforeend', getPortfolioHTML(new_pfo_idx));

    const new_pfo_el = pfo_div.querySelector(`#pfo_wrapper_${new_pfo_idx}`);
    new_pfo_el.querySelector('button.add_asset_btn').addEventListener('click', e => add_asset(new_pfo_idx));
    new_pfo_el.querySelector('button.delete_asset_btn').addEventListener('click', e => delete_asset(new_pfo_idx));   
}

function delete_portfolio() {
    const pfo_div = document.getElementById("pfo_div");

    if(pfo_div.children.length) {
        pfo_div.children[pfo_div.children.length-1].remove();
    }    
}

function add_asset(pfo_idx) {
    const pfo_el = document.body.querySelector(`#pfo_wrapper_${pfo_idx}`);
    const asset_list_el = pfo_el.querySelector('.asset_list');
    asset_list_el.insertAdjacentHTML('beforeend', getAssetHTML());
}

function delete_asset(pfo_idx) {
    const pfo_el = document.body.querySelector(`#pfo_wrapper_${pfo_idx}`);
    const asset_list_el = pfo_el.querySelector('.asset_list');
    const asset_list_length = pfo_el.querySelectorAll('.asset_list li').length;

    if(asset_list_length) {
        asset_list_el.children[asset_list_length-1].remove();
    }    
}