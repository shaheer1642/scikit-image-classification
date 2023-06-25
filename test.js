

fetch('https://w7.pngwing.com/pngs/817/902/png-transparent-google-logo-google-doodle-google-search-google-company-text-logo-thumbnail.png').then(async response => {
    const blob = await response.arrayBuffer();
    const contentType = response.headers.get('content-type');
    const base64String = `${Buffer.from(
      blob,
    ).toString('base64')}`;

    console.log(base64String)
    
    fetch('https://maize-detection-api.up.railway.app/predict', {
        method: 'POST',
        headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json'
        },
        body: JSON.stringify({image: base64String})
    }).then(res => res.json()).then(res => {
        console.log(res)
    })
}).catch(console.error)
