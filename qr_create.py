import qrcode

url = "https://github.com/dajuarez4/Giant-impact-hypothesis-and-formation-of-the-Moon"
img = qrcode.make(url)
img.save("./qr_github.png")
print("Saved: qr_github.png")

