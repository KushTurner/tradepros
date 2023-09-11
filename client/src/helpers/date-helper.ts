export const convertDateToUnixTimestamp = (date: Date) => {
  return Math.floor(date.getTime() / 1000);
};

export const convertUnixTimestampToDateTime = (
  unixTimestamp: number,
  resolution: string
) => {
  const milliseconds = unixTimestamp * 1000;
  const date = new Date(milliseconds);

  const day = date.getDate();
  const month = date.toLocaleString('default', { month: 'short' });
  const year = date.getFullYear();
  const hours = date.getHours().toString().padStart(2, '0');
  const minutes = date.getMinutes().toString().padStart(2, '0');

  if (resolution === '1Y') {
    return `${day} ${month} ${year}`;
  }
  return `${day} ${month} ${year} ${hours}:${minutes}`;
};

export const createDate = (
  date: Date,
  days: number,
  weeks: number,
  months: number,
  years: number
) => {
  const newDate = new Date(date);
  newDate.setDate(newDate.getDate() + days + 7 * weeks);
  newDate.setMonth(newDate.getMonth() + months);
  newDate.setFullYear(newDate.getFullYear() + years);
  return newDate;
};

export const todaysDate = () => {
  const today = new Date();
  const year = today.getFullYear();
  const month = (today.getMonth() + 1).toString().padStart(2, '0'); // Months are 0-based, so add 1
  const day = today.getDate().toString().padStart(2, '0');

  const formattedDate = `${year}-${month}-${day}`;
  return formattedDate;
};
