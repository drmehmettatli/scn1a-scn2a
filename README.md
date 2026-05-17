# scn1a-scn2a

SCN ailesi sodyum kanal proteinleri (SCN1A/2A/3A/4A/5A/7A/8A/9A/10A/11A) için UniProt üzerinden hizalama (alignment) yapıp benzerlik oranı (% identity) çıkarmak için uygulanabilir rehber.

## Amaç
Hocanın istediği şekilde:
1. UniProt’tan ilgili insan proteinlerini çekmek,
2. Çoklu hizalama (MSA) yapmak,
3. Pairwise benzerlik oranlarını (% identity) tabloya dökmek,
4. Kısa bir rapor hazırlamak.

---

## 1) UniProt’ta proteinleri topla

UniProt arama kutusuna aşağıdaki sorguları tek tek yapıştır:

- `gene:SCN1A AND organism_id:9606 AND reviewed:true`
- `gene:SCN2A AND organism_id:9606 AND reviewed:true`
- `gene:SCN3A AND organism_id:9606 AND reviewed:true`
- `gene:SCN4A AND organism_id:9606 AND reviewed:true`
- `gene:SCN5A AND organism_id:9606 AND reviewed:true`
- `gene:SCN7A AND organism_id:9606 AND reviewed:true`
- `gene:SCN8A AND organism_id:9606 AND reviewed:true`
- `gene:SCN9A AND organism_id:9606 AND reviewed:true`
- `gene:SCN10A AND organism_id:9606 AND reviewed:true`
- `gene:SCN11A AND organism_id:9606 AND reviewed:true`

### Seçim kriteri
- **Organism:** Homo sapiens
- **Reviewed:** Swiss-Prot
- **Sequence:** **Canonical isoform**

### FASTA indirme
- Kayıtları seç (10 protein)
- **Download → FASTA (canonical)**
- Tek bir FASTA dosyası olarak kaydet (örn. `scn_channels_human_canonical.fasta`)

---

## 2) Çoklu hizalama (MSA)

İki pratik seçenek:

### Seçenek A — UniProt Align (önerilen)
1. UniProt’ta seçili kayıtlarla **Align** aracını aç.
2. İndirilen FASTA’yı yükle.
3. Varsayılan parametrelerle çalıştır.
4. Çıktıdan şunları indir/kaydet:
   - MSA sonucu (alignment)
   - Pairwise/percent identity matrisi (varsa)
   - Guide tree (varsa)

### Seçenek B — EBI Clustal Omega
1. https://www.ebi.ac.uk/Tools/msa/clustalo/
2. FASTA dosyasını yapıştır/yükle.
3. Varsayılan ayarlarla çalıştır.
4. Sonuçlardan alignment ve percent identity çıktısını al.

---

## 3) Benzerlik oranlarını (% identity) çıkar

10 protein için toplam **45 farklı çift** vardır.

### Matrisi doldurma formatı
- Satır ve sütunlar: SCN1A, SCN2A, SCN3A, SCN4A, SCN5A, SCN7A, SCN8A, SCN9A, SCN10A, SCN11A
- Hücre: ilgili iki protein arası `% identity`
- Köşegen (aynı protein): `%100`

Ayrıca şu iki bilgiyi rapora ekle:
- **En yüksek benzerlik** görülen çift(ler)
- **En düşük benzerlik** görülen çift(ler)

---

## 4) Hızlı rapor şablonu (hocaya teslim)

Aşağıdaki metni doldur:

### Yöntem
"Homo sapiens için SCN1A, SCN2A, SCN3A, SCN4A, SCN5A, SCN7A, SCN8A, SCN9A, SCN10A ve SCN11A proteinlerinin canonical dizileri UniProt (Reviewed/Swiss-Prot) üzerinden FASTA formatında alındı. Diziler UniProt Align / Clustal Omega ile çoklu hizalandı. Pairwise yüzde benzerlik (% identity) matrisi çıkarıldı."

### Sonuç
- 10x10 benzerlik matrisi eklendi.
- En yüksek benzerlik: `...`
- En düşük benzerlik: `...`

### Kısa yorum
- **SCN7A (NaX)** atipik olduğu için çoğu zaman diğer klasik Nav kanallarına göre daha düşük benzerlik gösterebilir.

---

## 5) Kritik bilimsel not (önerilen ikinci analiz)

Full-length protein karşılaştırması yapılırken değişken bölgeler sonuçları etkileyebilir.
Daha biyolojik anlamlı bir karşılaştırma için ikinci turda sadece korunmuş kanal bölgeleri (özellikle transmembran/pore domainleri) ile yeniden hizalama yapılması önerilir.

---

## 6) Teslim checklist

- [ ] 10 proteinin canonical FASTA’sı indirildi
- [ ] MSA çalıştırıldı
- [ ] Percent identity matrisi çıkarıldı
- [ ] En yüksek/en düşük çiftler yazıldı
- [ ] Kısa yöntem + sonuç + yorum raporu tamamlandı
