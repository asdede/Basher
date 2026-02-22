# Päiväkirja projektiin liittyen
*Poistetaan*

## 22.2.26

- Lisätty virheiden hallitaa
- Lisätty tarkempaa dokumentaation .sh scriptiin
- Lisätty uusi endpoint /health
- Lueskelua RAG-koulutuksesta kielimallille.

## 14.7.26

### Ohjelman työstöä
- Lisätty webscaperit hakemaan eilen päätetyt dokumentaatiot. Fedoran quick-docs ladattu koneelle reposta /docs kansioon.
- Onnistuneesti lisätty kaikki tarvittava piene taistelun jälkeen. Jouduttu tekemään uniikit metodit sivujen lukemiseen.
- Yhdistetty kielimalli ja api
- Yhdistetty kielimalli käyttämään databasea apunaan
- Testausta ja testausta. Aika hyvin näyttäisi jo toimivan. Vielä muutamia bugeja. Nyt vain (ask mode)
- Muutamat issuet tehty tulevaisuuden suunnitelmista ja bugista
    - Terminaali kaatuu, jos käyttää komentoja joita ei ole vielä asennettu kun 'basheriä' käyttää
    - Agent-modista issue
    - Muistista issue. Tavoitteena tehdä persoonallisempi AI-avustaja. Mietinnässä vielä, mitä tietoja tänne laittaisia ja miten tämä toimisi.
    - .sh tiedoston lisäyksistä.
- Scripti tehty, jolla saa käytettyä basheriä. `basher hello world whats ypp`


## 13.7.26
Tänään saatu idea [tästä](https://fedoramagazine.org/how-to-make-a-local-open-source-ai-chatbot-who-has-access-to-fedora-documentation) postauksesta 

### Suunnitelman tekoa

Suunnitelmana on tehdä ai avustaja terminaaliin helpottamaan yleisesti linuxin käyttöä.

Tutkittu erilaisia tietokantoja, jotka soveltuvat kielimallille. Tehty päätös käyttää Qdrant-tietokantaa.

Päätetty lisätä seuraavat dokumentaatiot kielimallille käytettäväksi:
- Fedora docs
- Manual (man) ref
- Bash scripting ref
- Docker cli docs

Näillä pääsee aika hyvin jo alkuun, ehkä tulevaisuudessa jotain lisää?

- Dockerfilet luotu ja docker-compose.yml. Saatu koko stydeemi kasaan.

Arkkitehtuuri on siis seuraava:

1. API-kontti
2. DB-kontti
3. Kielimalli-kontti

Apia kutsutaan terminaalista ja hoitaa hommat databasen ja kielimallin kanssa.