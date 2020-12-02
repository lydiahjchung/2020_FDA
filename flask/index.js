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
                    <input type="text" class="asset asset_tiker" name="asset_tiker" placeholder="Ticker"/>
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
            <input type="text" class="asset asset_tiker" name="asset_tiker" placeholder="Ticker"/>
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

function input_data_to_array() {
    const start_date = document.body.querySelector('.start_date').value;
    const end_date = document.body.querySelector('.end_date').value;

    const pfo_data = [];
    const pfo_div = [...document.getElementById("pfo_div").children].forEach(el => {
        const pfo_name = el.querySelector('.pfo_name').value;

        const assets = [];
        [...el.querySelector('.asset_list').children].forEach(el => {
            const tiker = el.querySelector('input.asset_tiker').value;
            const name = el.querySelector('input.asset_name').value;
            const weight = el.querySelector('input.asset_weight').value;
            const leverage = el.querySelector('input.asset_leverage').value;

            assets.push({
                tiker,
                asset_name: name,
                weight,
                leverage
            });
        });

        pfo_data.push({
            profile_name: pfo_name,
            assets
        })
    });


    var pfos = {
        start_date, 
        end_date,
        profiles: pfo_data
    };

    $.post("/post", pfos).done(function(json) {
        console.log(json) 
    }).fail(function(xhr, status, errorThrown) {
        console.log(errorThrown) 
    });

    // });

    return {
        start_date, 
        end_date,
        profiles: pfo_data
    };
}