let cart = [];

function addToCart(name, price) {
    let existingItem = cart.find(item => item.name === name);
    if (existingItem) {
        existingItem.quantity++;
    } else {
        cart.push({ name, price, quantity: 1 });
    }
    updateCart();
}

function updateCart() {
    let cartItems = document.getElementById("cartItems");
    let subtotal = 0;

    cartItems.innerHTML = ""; // Clear previous items

    cart.forEach(item => {
        let li = document.createElement("li");
        li.textContent = `${item.name} - ₹${item.price}`;
        cartItems.appendChild(li);
        subtotal += item.price;
    });

    let gst = subtotal * 0.18;
    let discount = subtotal > 500 ? subtotal * 0.1 : 0;
    let total = subtotal + gst - discount;

    document.getElementById("subtotal").textContent = subtotal.toFixed(2);
    document.getElementById("gst").textContent = gst.toFixed(2);
    document.getElementById("discount").textContent = discount.toFixed(2);
    document.getElementById("total").textContent = total.toFixed(2);
}

function removeFromCart(name) {
    cart = cart.filter(item => item.name !== name);
    updateCart();
}

function filterMedicines() {
    let searchValue = document.getElementById("searchInput").value.toLowerCase();
    let medicines = document.querySelectorAll(".medicine-card");

    medicines.forEach(card => {
        let name = card.querySelector("h3").innerText.toLowerCase();
        card.style.display = name.includes(searchValue) ? "block" : "none";
    });
}

function clearCart() {
    fetch('/clear_cart', { method: 'POST' })  // Call Flask route
    .then(response => window.location.reload());  // Reload the page to reflect changes
}



function checkout() {
    alert("Proceeding to checkout!");
}


