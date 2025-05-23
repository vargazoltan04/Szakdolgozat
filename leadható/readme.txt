README.txt
==========

1. Projekt neve
---------------
Optikai karakterfelismerés konvolúciós neurális hálózatokkal

2. A projekt célja
-------------------
Ez a konzolos alkalmazás képekből nyeri ki az ott található szöveget, majd 
– amennyiben a felhasználó biztosítja – összeveti az eredeti szöveggel, és statisztikai értékelést végez. 
A szoftver csak .png és .jpg képekkel dolgozik

3. Rendszerkövetelmények
-------------------------
- Python 3.12.7 vagy újabb
- Windows 10 operációs rendszer
- Internetkapcsolat a függőségek első telepítéséhez

4. Használt könyvtárak
-----------------------
A következő főbb Python-csomagokat használja a program (részletezve a 'requirements.txt' fájlban):

- 'textdistance'
- 'numpy'
- 'pathlib'
- 'opencv-python'
- 'matplotlib'
- 'seaborn'
- 'Levenshtein'
- 'pytorch'
- 'torchvision'
- 'PIL'
- 'scikit-learn'

Telepítéskor a 'requirements.txt' fájl alapján ezek automatikusan telepítésre kerülnek.

5. Telepítés és futtatás
-------------------------
1. Hozz létre egy virtuális környezetet a projekt főkönyvtárában:
A szoftver egy konzolos alkalmazás, mely Python nyelven írodott. 
A használatához szükséges telepíteni a felhasznált Python verziót, illetve a felhasznált függvénykönyvtárakat. 
A program futtatásához a felhasznált csomagokat telepíteni a következő parancsok kiadásával lehet a projekt főkönyvtárában: 
1.	python -m venv .
2.	source Scripts/activate
3.	pip install -r requirements.txt
A program a main.py fájl futtatásával indítható el.

A neurális háló súlyokat a következő linken lehet elérni:
https://drive.google.com/file/d/13w3RfT4jpOiOgm7ZAKMAVcAj56TJsCfM/view?usp=sharing

Ezt letöltés után az ./src/network mappába kell másolni a megfelelő működés érdekében.



6. Parancssori kapcsolók
--------------------------
--path <Fájl elérési útvonala>: A bemeneti kép elérési útvonala. Mindenképpen egy képfájl kell legyen. 
				A <> jeleket nem kell megadni az elérési útvonalhoz. 
				Alternatívája a -p. Ha a –help kapcsoló nem létezik, akkor kötelező megadni. 
--output_path <Elérési útvonal>: Ebbe a mappába menti a program a kimenetet, és a részeredményeket. Ha nem létezik a mappa, akkor a program létrehozza azt. 
				 A <> jeleket nem kell megadni az elérési útvonalhoz. Nem kötelező megadni, ha nincs megadva, akkor a kimenetet a bemeneti kép mellé menti. 
				 Alternatívája a -op. 
--help: Segítséget nyújt a program használatához. Nem kötelező megadni. 
--debug: Amennyiben létezik ez a kapcsoló, akkor a program elmenti a részeredményeket is. 
	 Ha meg van adva, akkor a bemeneti képpel megegyező nevű .txt kiterjesztésű fileban meg kell adni a képen lévő eredeti szöveget, 
	 hogy össze tudja hasonlítani vele és kiszámolni a különféle statisztikákat a program. Ha nem létezik, akkor a kimenet csak egy szöveges fájlból áll. Nem kötelező megadni. 

7. Kimenet
-----------
A program eredményként egy szövegfájlt generál, amely a képen található szöveget tartalmazza. 
Debug üzemmódban további fájlok is keletkeznek, például közbenső képek, illetve egy statisztikai elemzés a felismerés pontosságáról.

8. Ismert korlátozások
------------------------
- Csak képfájlokat fogad el bemenetként (.jpg és .png formátumban)
- Jelenleg csak az angol abc kis és nagybetűit ismeri fel, illetve néhány speciális karaktert
- A referencia szövegnek pontos formázásban kell egyeznie a képen lévő szöveggel

9. Fejlesztési lehetőségek
---------------------------
- Felhasználói felület (GUI) hozzáadása
- PDF vagy többoldalas bemenet támogatása
- További statisztikai mutatók számítása (pl. karakter- vagy szóalapú hibaarány)

10. Készítő
------------
- Név: Varga Zoltán
- Neptun kód: QACW3P	
- Intézmény: Szegedi Tudományegyetem Természettudományi és Informatikai Kar
- Konzulens: Dr. Palágyi Kálmán
- Dátum: 2025.05

