# خطوات رفع المستودع إلى GitHub

## اسم المستودع المقترح

```text
5G-NIDD-Leakage-Aware-IDS
```

أنشئ مستودعًا عامًا فارغًا في GitHub، ولا تطلب من GitHub إنشاء README أو License أو `.gitignore`؛ لأن هذه الملفات موجودة بالفعل في الحزمة.

## الرفع باستخدام Git Bash أو Terminal

افتح الطرفية داخل مجلد المشروع، ثم نفذ:

```bash
git init
git add .
git commit -m "Initial reproducibility release"
git branch -M main
git remote add origin https://github.com/USERNAME/5G-NIDD-Leakage-Aware-IDS.git
git push -u origin main
```

استبدل `USERNAME` باسم حسابك في GitHub.

## فحص مهم بعد الرفع

تأكد من عدم ظهور الملفات أو المجلدات الآتية في GitHub:

```text
data/Combined.csv
work/
results/
```

ملف البيانات محمي بواسطة `.gitignore` ولا ينبغي رفعه أو توزيعه داخل المستودع.

## إصدار النسخة الأولى

بعد التأكد من المستودع، أنشئ Release بالوسم:

```text
v1.0.0
```

ثم اربط المستودع بـZenodo للحصول على DOI ثابت للكود، وضع DOI الناتج في فقرة Code Availability داخل الورقة.
